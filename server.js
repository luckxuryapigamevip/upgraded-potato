const express = require('express');
const axios = require('axios');
const fs = require('fs');
const readline = require('readline');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 5000;

app.use(cors());
app.use(express.json());

// ==================== CONFIG ====================
const API_URL_HU = 'https://wtx.tele68.com/v1/tx/sessions';
const API_URL_MD5 = 'https://wtxmd52.tele68.com/v1/txmd5/sessions';
const LEARNING_FILE = 'tiendat.json';
const HISTORY_FILE = 'tiendat1.json';
const DUDOAN_HISTORY_FILE = 'dudoan_history.json';
const DATA_17K_FILE = 'data_sessions.csv';
const PATTERN_CAU_FILE = 'cau.txt';
const TRAINED_MODEL_FILE = 'trained_model_17k.json';

const MAX_HISTORY = 100;
const MAX_DUDOAN_HISTORY = 1000;
const AUTO_SAVE_INTERVAL = 30000;
const AUTHOR_ID = '@tranhoang2286';

// ==================== STATE ====================
let predictionHistory = { hu: [], md5: [] };
let dudoanHistory = { hu: [], md5: [] };
let lastProcessedPhien = { hu: null, md5: null };
let lastPattern10 = { hu: [], md5: [] };

// TRẠNG THÁI CẦU (QUAN TRỌNG)
// mode: 'bam_bet' | 'pha_cau' | 'bam_dao' | 'theo_dong'
// breakState: theo dõi khi cầu vừa gãy
let cauState = {
  hu: { mode: 'theo_dong', streakType: null, streakLen: 0, lastBroken: false, breakCount: 0, breakPhien: null, breakPrediction: null },
  md5: { mode: 'theo_dong', streakType: null, streakLen: 0, lastBroken: false, breakCount: 0, breakPhien: null, breakPrediction: null }
};

let trainedModel = {
  trained: false, totalSamples: 0, markovChains: {}, patternDict: {}, taiPercent: 0, xiuPercent: 0
};

let learningData = {
  hu: { predictions: [], totalPredictions: 0, correctPredictions: 0 },
  md5: { predictions: [], totalPredictions: 0, correctPredictions: 0 }
};

// ==================== FILE I/O ====================
function loadLearningData() {
  try { if (fs.existsSync(LEARNING_FILE)) learningData = { ...learningData, ...JSON.parse(fs.readFileSync(LEARNING_FILE, 'utf8')) }; } catch (e) {}
}
function saveLearningData() {
  try { fs.writeFileSync(LEARNING_FILE, JSON.stringify(learningData, null, 2)); } catch (e) {}
}
function loadPredictionHistory() {
  try {
    if (fs.existsSync(HISTORY_FILE)) {
      const p = JSON.parse(fs.readFileSync(HISTORY_FILE, 'utf8'));
      predictionHistory = p.history || { hu: [], md5: [] };
      lastProcessedPhien = p.lastProcessedPhien || { hu: null, md5: null };
      cauState = p.cauState || cauState;
    }
    if (fs.existsSync(DUDOAN_HISTORY_FILE)) {
      const p2 = JSON.parse(fs.readFileSync(DUDOAN_HISTORY_FILE, 'utf8'));
      dudoanHistory = p2.history || { hu: [], md5: [] };
    }
  } catch (e) {}
}
function savePredictionHistory() {
  try {
    fs.writeFileSync(HISTORY_FILE, JSON.stringify({ history: predictionHistory, lastProcessedPhien, cauState }, null, 2));
    fs.writeFileSync(DUDOAN_HISTORY_FILE, JSON.stringify({ history: dudoanHistory }, null, 2));
  } catch (e) {}
}

// ==================== MODEL ====================
function parseCSVLine(line) {
  const parts = line.split(','); if (parts.length < 5) return null;
  return { Phien: parseInt(parts[0]), Ket_qua: parts[1].trim() === 'TAI' ? 'Tài' : 'Xỉu' };
}
function buildMarkovChainFromArray(results, order) {
  const chain = {};
  for (let i = 0; i <= results.length - order - 1; i++) {
    const state = results.slice(i, i + order).map(r => r === 'Tài' ? 'T' : 'X').join('');
    const next = results[i + order];
    if (!chain[state]) chain[state] = { 'Tài': 0, 'Xỉu': 0, total: 0 };
    chain[state][next]++; chain[state].total++;
  }
  return chain;
}
function predictMarkovFromModel(results, order) {
  const chain = trainedModel.markovChains[order]; if (!chain || results.length < order) return null;
  const currentState = results.slice(0, order).map(r => r === 'Tài' ? 'T' : 'X').join('');
  const stateData = chain[currentState];
  if (!stateData || stateData.total < 5) return null;
  const taiProb = stateData['Tài'] / stateData.total, xiuProb = stateData['Xỉu'] / stateData.total;
  const confidence = Math.max(taiProb, xiuProb);
  if (confidence < 0.55) return null;
  return { prediction: taiProb > xiuProb ? 'Tài' : 'Xỉu', confidence: Math.round(confidence * 100), order };
}
function predictFromPatternDict(resultsArray) {
  if (resultsArray.length < 14) return null;
  const currentPattern = resultsArray.slice(0, 14).reverse().map(r => r === 'Tài' ? 'T' : 'X').join('');
  if (trainedModel.patternDict && trainedModel.patternDict[currentPattern]) {
    return { prediction: trainedModel.patternDict[currentPattern] === 'T' ? 'Tài' : 'Xỉu', name: `Pattern Match (${currentPattern})` };
  }
  return null;
}
async function loadAndTrainFromCSV() {
  if (!fs.existsSync(DATA_17K_FILE)) return false;
  try {
    const rl = readline.createInterface({ input: fs.createReadStream(DATA_17K_FILE), crlfDelay: Infinity });
    let allSessions = [], headerSkipped = false;
    for await (const line of rl) {
      if (!line.trim()) continue;
      if (!headerSkipped && line.toLowerCase().includes('dsid')) { headerSkipped = true; continue; }
      headerSkipped = true;
      const parsed = parseCSVLine(line);
      if (parsed) allSessions.push(parsed);
    }
    allSessions.sort((a, b) => b.Phien - a.Phien);
    const results = allSessions.map(s => s.Ket_qua);
    for (let order = 1; order <= 9; order++) trainedModel.markovChains[order] = buildMarkovChainFromArray(results, order);
    trainedModel.totalSamples = allSessions.length; trainedModel.trained = true;
    fs.writeFileSync(TRAINED_MODEL_FILE, JSON.stringify(trainedModel));
    return true;
  } catch (e) { return false; }
}
function loadTrainedModel() {
  try { if (fs.existsSync(TRAINED_MODEL_FILE)) { trainedModel = JSON.parse(fs.readFileSync(TRAINED_MODEL_FILE, 'utf8')); return true; } } catch (e) {}
  return false;
}
async function loadPatternCau() {
  if (!fs.existsSync(PATTERN_CAU_FILE)) return;
  const rl = readline.createInterface({ input: fs.createReadStream(PATTERN_CAU_FILE), crlfDelay: Infinity });
  for await (const line of rl) {
    const match = line.match(/([TX]+)\s*-\s*([TX])/);
    if (match && match[1].length === 14) trainedModel.patternDict[match[1].trim()] = match[2].trim();
  }
}

// ==================== PHÂN TÍCH CẦU (TRÁI TIM MỚI) ====================

/**
 * Phát hiện loại cầu từ mảng kết quả
 * Returns: { type, len, value }
 * type: 'streak' | 'alternating' | 'mixed'
 */
function detectCau(results) {
  if (!results || results.length < 2) return { type: 'mixed', len: 0, value: null };

  // Kiểm tra bệt
  let streakLen = 1;
  for (let i = 1; i < results.length; i++) {
    if (results[i] === results[0]) streakLen++;
    else break;
  }
  if (streakLen >= 3) return { type: 'streak', len: streakLen, value: results[0] };

  // Kiểm tra đảo 1-1
  let altLen = 1;
  for (let i = 1; i < results.length; i++) {
    if (results[i] !== results[i - 1]) altLen++;
    else break;
  }
  if (altLen >= 4) return { type: 'alternating', len: altLen, value: results[0] };

  return { type: 'mixed', len: 0, value: null };
}

/**
 * Phát hiện gãy cầu: so sánh cầu vừa xong với dự đoán trước
 * Nếu cầu bệt đang chạy mà xuất hiện 1 con khác -> gãy 1 nháy
 * Nếu cầu đảo đang chạy mà xuất hiện 2 con cùng -> gãy
 * 
 * Logic bẻ cầu gãy:
 * - Cầu bệt gãy 1 nháy: phiên tiếp theo đánh CÙNG loại con bị gãy (bẻ lại về cầu cũ)
 * - Cầu đảo gãy (2 con cùng loại): phiên tiếp theo đánh CÙNG loại con xuất hiện 2 lần đó
 */
function analyzeBreak(results, state) {
  const r = results; // r[0] = mới nhất

  // --- PHÁT HIỆN GÃY CẦU BỆT ---
  // Trước đó có bệt >= 3, giờ thấy 1 con khác ở r[0]
  // r[0] khác r[1], r[1]=r[2]=r[3] (bệt cũ)
  if (r.length >= 4 && r[1] === r[2] && r[2] === r[3] && r[0] !== r[1]) {
    const oldStreak = r[1];
    const breakerType = r[0];
    // Đếm bệt cũ
    let oldLen = 0;
    for (let i = 1; i < r.length; i++) { if (r[i] === oldStreak) oldLen++; else break; }

    if (oldLen >= 2) {
      // GÃY CẦU BỆT: phiên tới bẻ theo con bị gãy (theo breakerType)
      return {
        isBreak: true,
        breakType: 'streak_break',
        prediction: breakerType, // đánh theo con vừa gãy vào (xu hướng mới)
        confidence: 72,
        reason: `Cầu bệt ${oldLen+1}x${oldStreak} vừa gãy bởi ${breakerType} -> bẻ theo ${breakerType}`
      };
    }
  }

  // --- PHÁT HIỆN GÃY CẦU ĐẢO (1-1) ---
  // Cầu đảo đang chạy: T-X-T-X-T...
  // Gãy khi: 2 con cùng loại liên tiếp sau chuỗi đảo
  // r[0]=r[1] (2 con cùng) nhưng r[2]!=r[1] (trước đó là đảo)
  if (r.length >= 5) {
    let altLenFromR2 = 1;
    for (let i = 3; i < Math.min(r.length, 10); i++) {
      if (r[i] !== r[i - 1]) altLenFromR2++;
      else break;
    }

    if (r[0] === r[1] && r[1] !== r[2] && altLenFromR2 >= 3) {
      const doubleType = r[0]; // con xuất hiện 2 lần
      // GÃY CẦU ĐẢO: 2 con cùng loại vừa xuất hiện -> xu hướng mới bắt đầu
      return {
        isBreak: true,
        breakType: 'alternating_break',
        prediction: doubleType, // đánh theo con double (cầu bệt mới có thể bắt đầu)
        confidence: 70,
        reason: `Cầu đảo ${altLenFromR2+2} phiên vừa gãy: 2x${doubleType} xuất hiện -> bẻ theo ${doubleType}`
      };
    }
  }

  // --- PHÁT HIỆN GÃY CẦU ĐẢO KIỂU 2: đang đảo T-X-T-X mà dự đoán sai ---
  // Trường hợp: cầu đang đảo, ta dự đoán T nhưng ra X-X (2 Xỉu liên tiếp)
  // -> phiên sau tiếp tục đánh X (xu hướng bệt mới)
  // Logic này đã nằm trong trường hợp trên

  return { isBreak: false };
}

// ==================== THUẬT TOÁN DỰ ĐOÁN CHÍNH ====================
function calculateAdvancedPrediction(data, type) {
  const results = data.slice(0, 50).map(d => d.Ket_qua);
  const state = cauState[type];

  // --- BƯỚC 1: PHÁT HIỆN GÃY CẦU ---
  const breakInfo = analyzeBreak(results, state);
  if (breakInfo.isBreak) {
    return {
      prediction: breakInfo.prediction,
      confidence: breakInfo.confidence,
      factors: [`🔄 BẺ CẦU GÃY: ${breakInfo.reason}`],
      mode: 'pha_cau'
    };
  }

  // --- BƯỚC 2: CẦU BỆT ---
  let streakLen = 1;
  for (let i = 1; i < results.length; i++) {
    if (results[i] === results[0]) streakLen++;
    else break;
  }
  const streakType = results[0];

  if (streakLen >= 3 && streakLen <= 7) {
    return {
      prediction: streakType,
      confidence: 65 + streakLen * 2,
      factors: [`🔥 BÁM BỆT ${streakLen}x${streakType} (Trend is friend)`],
      mode: 'bam_bet'
    };
  }

  // Cầu rồng >= 8: bắt đầu bẻ
  if (streakLen >= 8) {
    return {
      prediction: streakType === 'Tài' ? 'Xỉu' : 'Tài',
      confidence: 75,
      factors: [`🐉 BẺ CẦU RỒNG ${streakLen}x${streakType}`],
      mode: 'pha_cau'
    };
  }

  // --- BƯỚC 3: CẦU ĐẢO 1-1 ---
  let altLen = 1;
  for (let i = 1; i < results.length; i++) {
    if (results[i] !== results[i - 1]) altLen++;
    else break;
  }
  if (altLen >= 4) {
    const nextExpected = results[0] === 'Tài' ? 'Xỉu' : 'Tài';
    return {
      prediction: nextExpected,
      confidence: 65 + altLen,
      factors: [`🔄 BÁM ĐẢO 1-1 (${altLen} phiên), tiếp theo: ${nextExpected}`],
      mode: 'bam_dao'
    };
  }

  // --- BƯỚC 4: MARKOV 17K ---
  if (trainedModel.trained) {
    for (let order = 9; order >= 5; order--) {
      const p = predictMarkovFromModel(results, order);
      if (p && p.confidence >= 70) {
        return {
          prediction: p.prediction,
          confidence: p.confidence,
          factors: [`🧠 MARKOV BẬC ${order} (17K mẫu) -> ${p.prediction}`],
          mode: 'markov'
        };
      }
    }
  }

  // --- BƯỚC 5: PATTERN 10K ---
  const pd = predictFromPatternDict(results);
  if (pd) {
    return { prediction: pd.prediction, confidence: 85, factors: [`📊 PATTERN 10K: ${pd.name}`], mode: 'pattern' };
  }

  // --- BƯỚC 6: FALLBACK SỐ ĐÔNG ---
  const tc = results.slice(0, 10).filter(r => r === 'Tài').length;
  return {
    prediction: tc >= 5 ? 'Tài' : 'Xỉu',
    confidence: 55,
    factors: [`📉 Không rõ cầu - Số đông 10 phiên (Tài:${tc}/10)`],
    mode: 'fallback'
  };
}

// ==================== API FETCH ====================
function transformApiData(apiData) {
  if (!apiData || !apiData.list || !Array.isArray(apiData.list)) return null;
  return apiData.list.map(item => ({
    Phien: item.id, Ket_qua: item.resultTruyenThong === 'TAI' ? 'Tài' : 'Xỉu',
    Xuc_xac_1: item.dices[0], Xuc_xac_2: item.dices[1], Xuc_xac_3: item.dices[2], Tong: item.point
  }));
}
async function fetchDataHu() { try { return transformApiData((await axios.get(API_URL_HU, { timeout: 10000 })).data); } catch (e) { return null; } }
async function fetchDataMd5() { try { return transformApiData((await axios.get(API_URL_MD5, { timeout: 10000 })).data); } catch (e) { return null; } }

function updatePattern10(type, data) {
  lastPattern10[type] = data.slice(0, 10).map(d => d.Ket_qua === 'Tài' ? 'T' : 'X');
  return lastPattern10[type].join('');
}

// ==================== LƯU LỊCH SỬ ====================
function saveDudoanRecord(type, phienTruoc, resultTruoc, phienNay, duDoan, doTinCay) {
  const rec = {
    phien_truoc: phienTruoc, result: resultTruoc, Phien_nay: phienNay,
    Du_doan: duDoan, Do_tin_cay: doTinCay, Trang_thai: '', timestamp: new Date().toISOString()
  };
  dudoanHistory[type].unshift(rec);
  if (dudoanHistory[type].length > MAX_DUDOAN_HISTORY) dudoanHistory[type].pop();
  return rec;
}

function savePredictionToHistory(type, phien, prediction, confidence, latestData, mode) {
  const rec = {
    Phien: latestData.Phien, Xuc_xac_1: latestData.Xuc_xac_1, Xuc_xac_2: latestData.Xuc_xac_2,
    Xuc_xac_3: latestData.Xuc_xac_3, Tong: latestData.Tong, Ket_qua: latestData.Ket_qua,
    Do_tin_cay: `${confidence}%`, Phien_hien_tai: phien.toString(),
    Du_doan: prediction, ket_qua_du_doan: '', Mode: mode || 'unknown',
    id: AUTHOR_ID, timestamp: new Date().toISOString()
  };
  predictionHistory[type].unshift(rec);
  if (predictionHistory[type].length > MAX_HISTORY) predictionHistory[type].pop();
  return rec;
}

async function updateHistoryStatus(type) {
  try {
    const data = type === 'hu' ? await fetchDataHu() : await fetchDataMd5();
    if (!data || data.length === 0) return;
    for (let i = 0; i < dudoanHistory[type].length; i++) {
      let rec = dudoanHistory[type][i];
      if (rec.Trang_thai === '') {
        const actual = data.find(d => d.Phien.toString() === rec.Phien_nay.toString());
        if (actual) {
          rec.Trang_thai = rec.Du_doan === actual.Ket_qua ? 'ĐÚNG ✅' : 'SAI ❌';
        }
      }
    }
    for (const rec of predictionHistory[type]) {
      if (rec.ket_qua_du_doan && rec.ket_qua_du_doan !== '') continue;
      const actual = data.find(d => d.Phien.toString() === rec.Phien_hien_tai);
      if (actual) rec.ket_qua_du_doan = rec.Du_doan === actual.Ket_qua ? 'Đúng ✅' : 'Sai ❌';
    }
    savePredictionHistory();
  } catch (e) {}
}

async function autoProcessPredictions() {
  try {
    const dataHu = await fetchDataHu();
    if (dataHu && dataHu.length > 0) {
      const nextPhien = dataHu[0].Phien + 1;
      if (lastProcessedPhien.hu !== nextPhien) {
        await updateHistoryStatus('hu');
        updatePattern10('hu', dataHu);
        const result = calculateAdvancedPrediction(dataHu, 'hu');
        cauState.hu.mode = result.mode;
        savePredictionToHistory('hu', nextPhien, result.prediction, result.confidence, dataHu[0], result.mode);
        saveDudoanRecord('hu', dataHu[0].Phien, dataHu[0].Ket_qua, nextPhien, result.prediction, `${result.confidence}%`);
        lastProcessedPhien.hu = nextPhien;
        console.log(`[HU] Phiên ${nextPhien}: ${result.prediction} (${result.confidence}%) [${result.mode}] - ${result.factors[0]}`);
      }
    }
    const dataMd5 = await fetchDataMd5();
    if (dataMd5 && dataMd5.length > 0) {
      const nextPhien = dataMd5[0].Phien + 1;
      if (lastProcessedPhien.md5 !== nextPhien) {
        await updateHistoryStatus('md5');
        updatePattern10('md5', dataMd5);
        const result = calculateAdvancedPrediction(dataMd5, 'md5');
        cauState.md5.mode = result.mode;
        savePredictionToHistory('md5', nextPhien, result.prediction, result.confidence, dataMd5[0], result.mode);
        saveDudoanRecord('md5', dataMd5[0].Phien, dataMd5[0].Ket_qua, nextPhien, result.prediction, `${result.confidence}%`);
        lastProcessedPhien.md5 = nextPhien;
        console.log(`[MD5] Phiên ${nextPhien}: ${result.prediction} (${result.confidence}%) [${result.mode}] - ${result.factors[0]}`);
      }
    }
  } catch (e) { console.error('[Auto]:', e.message); }
}

function startAutoSaveTask() {
  setTimeout(() => autoProcessPredictions(), 5000);
  setInterval(() => autoProcessPredictions(), AUTO_SAVE_INTERVAL);
}

// ==================== ENDPOINTS ====================
app.get('/', (req, res) => {
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.send(`🔥 API Tài Xỉu v2 (Break Detection) - Trained: ${trainedModel.trained ? '✅' : '❌'}`);
});

app.get('/api/tx', async (req, res) => {
  try {
    const data = await fetchDataHu();
    if (!data || data.length === 0) return res.status(500).json({ error: 'Không lấy được dữ liệu' });
    await updateHistoryStatus('hu');
    updatePattern10('hu', data);
    const nextPhien = data[0].Phien + 1;
    const result = calculateAdvancedPrediction(data, 'hu');
    cauState.hu.mode = result.mode;
    const rec = savePredictionToHistory('hu', nextPhien, result.prediction, result.confidence, data[0], result.mode);
    saveDudoanRecord('hu', data[0].Phien, data[0].Ket_qua, nextPhien, result.prediction, `${result.confidence}%`);
    res.json({
      Phien: rec.Phien, Xuc_xac_1: rec.Xuc_xac_1, Xuc_xac_2: rec.Xuc_xac_2, Xuc_xac_3: rec.Xuc_xac_3,
      Tong: rec.Tong, Ket_qua: rec.Ket_qua, Do_tin_cay: rec.Do_tin_cay, Phien_hien_tai: rec.Phien_hien_tai,
      Du_doan: rec.Du_doan, pattern_10: lastPattern10.hu.join(''), ket_qua_du_doan: rec.ket_qua_du_doan || '',
      Mode: result.mode, id: rec.id, Phan_tich: result.factors
    });
  } catch (e) { res.status(500).json({ error: 'Lỗi server' }); }
});

app.get('/api/tx/md5', async (req, res) => {
  try {
    const data = await fetchDataMd5();
    if (!data || data.length === 0) return res.status(500).json({ error: 'Không lấy được dữ liệu' });
    await updateHistoryStatus('md5');
    updatePattern10('md5', data);
    const nextPhien = data[0].Phien + 1;
    const result = calculateAdvancedPrediction(data, 'md5');
    cauState.md5.mode = result.mode;
    const rec = savePredictionToHistory('md5', nextPhien, result.prediction, result.confidence, data[0], result.mode);
    saveDudoanRecord('md5', data[0].Phien, data[0].Ket_qua, nextPhien, result.prediction, `${result.confidence}%`);
    res.json({
      Phien: rec.Phien, Xuc_xac_1: rec.Xuc_xac_1, Xuc_xac_2: rec.Xuc_xac_2, Xuc_xac_3: rec.Xuc_xac_3,
      Tong: rec.Tong, Ket_qua: rec.Ket_qua, Do_tin_cay: rec.Do_tin_cay, Phien_hien_tai: rec.Phien_hien_tai,
      Du_doan: rec.Du_doan, pattern_10: lastPattern10.md5.join(''), ket_qua_du_doan: rec.ket_qua_du_doan || '',
      Mode: result.mode, id: rec.id, Phan_tich: result.factors
    });
  } catch (e) { res.status(500).json({ error: 'Lỗi server' }); }
});

app.get('/api/history/dudoan', async (req, res) => {
  try {
    const type = req.query.type || 'hu';
    await updateHistoryStatus(type);
    res.json({ type: type === 'hu' ? 'Tài Xỉu Hũ' : 'Tài Xỉu MD5', total: dudoanHistory[type].length, history: dudoanHistory[type].slice(0, 1000) });
  } catch (e) { res.status(500).json({ error: 'Lỗi server' }); }
});

// ==================== START ====================
loadLearningData();
loadPredictionHistory();

if (loadTrainedModel()) {
  if (!trainedModel.patternDict) loadPatternCau();
} else {
  loadAndTrainFromCSV().then(ok => { if (ok) loadPatternCau(); });
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`🚀 Server tại port ${PORT}`);
  startAutoSaveTask();
});
