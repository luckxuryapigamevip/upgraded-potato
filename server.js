const express = require('express');
const axios = require('axios');
const fs = require('fs');
const path = require('path');
const readline = require('readline');

const app = express();
const PORT = process.env.PORT || 5000;

// ==================== CONFIG ====================
const API_URL_HU = 'https://wtx.tele68.com/v1/tx/sessions';
const API_URL_MD5 = 'https://wtxmd52.tele68.com/v1/txmd5/sessions';
const LEARNING_FILE = 'tiendat.json';
const HISTORY_FILE = 'tiendat1.json';
const DUDOAN_HISTORY_FILE = 'dudoan_history.json';
const DATA_17K_FILE = 'data_sessions.csv';
const PATTERN_CAU_FILE = 'cau.txt';
const FORMULA_CAU_FILE = 'cong_thuc_cau.txt';
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

let trainedModel = {
  trained: false,
  trainStartTime: null,
  trainEndTime: null,
  totalSamples: 0,
  markovChains: { 1: {}, 2: {}, 3: {}, 4: {}, 5: {}, 6: {}, 7: {}, 8: {}, 9: {} },
  frequencyData: {},
  patternLibrary: {},
  patternDict: {},
  formulaMap: {},
  taiPercent: 0,
  xiuPercent: 0,
  transitions: {},
  history: []
};

let learningData = {
  hu: createEmptyLearningData(),
  md5: createEmptyLearningData()
};

function createEmptyLearningData() {
  return {
    predictions: [],
    patternStats: {},
    totalPredictions: 0,
    correctPredictions: 0,
    patternWeights: {},
    lastUpdate: null,
    streakAnalysis: { wins: 0, losses: 0, currentStreak: 0, bestStreak: 0, worstStreak: 0 },
    adaptiveThresholds: {},
    recentAccuracy: [],
    markovChains: {},
    frequencyData: {},
    pattern10History: []
  };
}

const DEFAULT_PATTERN_WEIGHTS = {
  'cau_bet': 1.0, 'cau_dao_11': 1.0, 'cau_22': 1.0, 'cau_33': 1.0,
  'cau_121': 1.0, 'cau_123': 1.0, 'cau_321': 1.0,
  'cau_nhay_coc': 1.0, 'cau_nhip_nghieng': 1.0, 'cau_3van1': 1.0,
  'cau_be_cau': 1.0, 'cau_chu_ky': 1.0, 'distribution': 1.0,
  'dice_pattern': 1.0, 'sum_trend': 1.0, 'edge_cases': 1.0,
  'momentum': 1.0, 'cau_tu_nhien': 1.0,
  'cau_rong': 1.0, 'smart_bet': 1.0,
  'break_streak': 1.0, 'alternating_break': 1.0,
  'double_pair_break': 1.0, 'triple_pattern': 1.0,
  'tong_phan_tich': 1.5, 'xu_huong_manh': 1.3, 'dao_chieu': 1.4,
  'markov_chain': 1.2, 'frequency_analysis': 1.1,
  'pattern_dict': 1.8, 'formula_map': 1.6
};

// ==================== FILE I/O ====================
function loadLearningData() {
  try {
    if (fs.existsSync(LEARNING_FILE)) {
      const data = fs.readFileSync(LEARNING_FILE, 'utf8');
      const parsed = JSON.parse(data);
      learningData = { ...learningData, ...parsed };
      console.log('✅ Learning data loaded from tiendat.json');
    }
  } catch (error) {
    console.error('Error loading learning data:', error.message);
  }
}

function saveLearningData() {
  try {
    fs.writeFileSync(LEARNING_FILE, JSON.stringify(learningData, null, 2));
  } catch (error) {
    console.error('Error saving learning data:', error.message);
  }
}

function loadPredictionHistory() {
  try {
    if (fs.existsSync(HISTORY_FILE)) {
      const data = fs.readFileSync(HISTORY_FILE, 'utf8');
      const parsed = JSON.parse(data);
      predictionHistory = parsed.history || { hu: [], md5: [] };
      lastProcessedPhien = parsed.lastProcessedPhien || { hu: null, md5: null };
      console.log('✅ Prediction history loaded');
    }
    if (fs.existsSync(DUDOAN_HISTORY_FILE)) {
      const data2 = fs.readFileSync(DUDOAN_HISTORY_FILE, 'utf8');
      const parsed2 = JSON.parse(data2);
      dudoanHistory = parsed2.history || { hu: [], md5: [] };
      console.log('✅ Du doan history loaded');
    }
  } catch (error) {
    console.error('Error loading prediction history:', error.message);
  }
}

function savePredictionHistory() {
  try {
    fs.writeFileSync(HISTORY_FILE, JSON.stringify({
      history: predictionHistory,
      lastProcessedPhien,
      lastSaved: new Date().toISOString()
    }, null, 2));
    fs.writeFileSync(DUDOAN_HISTORY_FILE, JSON.stringify({
      history: dudoanHistory,
      lastSaved: new Date().toISOString()
    }, null, 2));
  } catch (error) {
    console.error('Error saving prediction history:', error.message);
  }
}

// ==================== ĐỌC FILE 17K CSV ====================
function parseCSVLine(line) {
  const parts = [];
  let current = '';
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') inQuotes = !inQuotes;
    else if (char === ',' && !inQuotes) { parts.push(current); current = ''; }
    else current += char;
  }
  parts.push(current);
  if (parts.length < 5) return null;
  const dsid = parts[0].trim();
  const result = parts[1].trim();
  const dicesStr = parts[2].trim();
  const point = parseFloat(parts[3].trim());
  if (!dsid || !result) return null;
  let dices = [0, 0, 0];
  try { dices = JSON.parse(dicesStr); } catch (e) {
    const cleaned = dicesStr.replace(/[\[\]]/g, '');
    dices = cleaned.split(',').map(x => parseInt(x.trim()));
  }
  return {
    Phien: parseInt(dsid),
    Ket_qua: result === 'TAI' ? 'Tài' : 'Xỉu',
    Xuc_xac_1: dices[0] || 0, Xuc_xac_2: dices[1] || 0, Xuc_xac_3: dices[2] || 0,
    Tong: point || (dices[0] + dices[1] + dices[2])
  };
}

async function loadPatternCau() {
  if (!fs.existsSync(PATTERN_CAU_FILE)) return;
  console.log(`📖 Học 10k cầu từ ${PATTERN_CAU_FILE}...`);
  const rl = readline.createInterface({ input: fs.createReadStream(PATTERN_CAU_FILE), crlfDelay: Infinity });
  let count = 0;
  for await (const line of rl) {
    const match = line.match(/([TX]+)\s*-\s*([TX])/);
    if (match && match[1].length === 14) {
      trainedModel.patternDict[match[1].trim()] = match[2].trim();
      count++;
    }
  }
  console.log(`✅ Đã học ${count} patterns từ cau.txt`);
}

async function loadFormulaCau() {
  if (!fs.existsSync(FORMULA_CAU_FILE)) return;
  console.log(`📖 Học bảng quy đổi từ ${FORMULA_CAU_FILE}...`);
  const lines = fs.readFileSync(FORMULA_CAU_FILE, 'utf8').split('\n');
  let count = 0;
  for (const line of lines) {
    if (line.includes('=')) {
      const parts = line.split('=').map(s => s.trim());
      if (parts.length === 2) {
        if (!trainedModel.formulaMap[parts[0]]) trainedModel.formulaMap[parts[0]] = [];
        if (!trainedModel.formulaMap[parts[1]]) trainedModel.formulaMap[parts[1]] = [];
        if (!trainedModel.formulaMap[parts[0]].includes(parts[1])) trainedModel.formulaMap[parts[0]].push(parts[1]);
        if (!trainedModel.formulaMap[parts[1]].includes(parts[0])) trainedModel.formulaMap[parts[1]].push(parts[0]);
        count++;
      }
    }
  }
  console.log(`✅ Đã học ${count} công thức quy đổi cầu`);
}

async function loadAndTrainFromCSV() {
  console.log('\n📚 === BẮT ĐẦU HỌC 17K PHIÊN ===');
  trainedModel.trainStartTime = new Date().toISOString();
  if (!fs.existsSync(DATA_17K_FILE)) {
    console.log('⚠️  Không tìm thấy data_sessions.csv - Bỏ qua training');
    trainedModel.trained = false;
    return false;
  }
  try {
    const fileStream = fs.createReadStream(DATA_17K_FILE);
    const rl = readline.createInterface({ input: fileStream, crlfDelay: Infinity });
    let allSessions = [];
    let lineNum = 0;
    let headerSkipped = false;
    for await (const line of rl) {
      lineNum++;
      if (!line.trim()) continue;
      if (!headerSkipped && line.toLowerCase().includes('dsid')) { headerSkipped = true; continue; }
      headerSkipped = true;
      const parsed = parseCSVLine(line);
      if (parsed) allSessions.push(parsed);
      if (lineNum % 1000 === 0) console.log(`   📖 Đã đọc ${lineNum} dòng...`);
    }
    console.log(`✅ Đã đọc ${allSessions.length} phiên từ CSV`);
    allSessions.sort((a, b) => b.Phien - a.Phien);
    const results = allSessions.map(s => s.Ket_qua);
    console.log(`🔗 Building Markov Chains bậc 1-9...`);
    for (let order = 1; order <= 9; order++) {
      trainedModel.markovChains[order] = buildMarkovChainFromArray(results, order);
      console.log(`   ✅ Markov bậc ${order}: ${Object.keys(trainedModel.markovChains[order]).length} states`);
    }
    console.log(`📊 Building Frequency Data...`);
    const totalTai = results.filter(r => r === 'Tài').length;
    trainedModel.taiPercent = (totalTai / results.length * 100);
    trainedModel.xiuPercent = 100 - trainedModel.taiPercent;
    [10, 20, 50, 100, 500, 1000].forEach(w => {
      if (results.length >= w) {
        const win = results.slice(0, w);
        const t = win.filter(r => r === 'Tài').length;
        trainedModel.frequencyData[`window_${w}`] = {
          tai: t, xiu: w - t, taiPercent: t / w * 100, xiuPercent: (w - t) / w * 100
        };
      }
    });
    console.log(`   ✅ T:${trainedModel.taiPercent.toFixed(2)}% - X:${trainedModel.xiuPercent.toFixed(2)}%`);
    console.log(`🔄 Building Transitions...`);
    const transitions = { 'Tài': { 'Tài': 0, 'Xỉu': 0 }, 'Xỉu': { 'Tài': 0, 'Xỉu': 0 } };
    for (let i = 0; i < results.length - 1; i++) transitions[results[i]][results[i+1]]++;
    trainedModel.transitions = transitions;
    console.log(`🎯 Building Pattern Library...`);
    trainedModel.patternLibrary = buildPatternLibrary(results);
    console.log(`   ✅ ${Object.keys(trainedModel.patternLibrary).length} patterns detected`);
    trainedModel.history = allSessions.slice(0, 1000).map(s => ({ Phien: s.Phien, Ket_qua: s.Ket_qua, Tong: s.Tong }));
    trainedModel.totalSamples = allSessions.length;
    trainedModel.trained = true;
    trainedModel.trainEndTime = new Date().toISOString();
    fs.writeFileSync(TRAINED_MODEL_FILE, JSON.stringify(trainedModel, null, 2));
    console.log(`\n🎉 === HOÀN TẤT HỌC ${allSessions.length} PHIÊN ===\n`);
    return true;
  } catch (error) {
    console.error('❌ Error training from CSV:', error.message);
    trainedModel.trained = false;
    return false;
  }
}

function loadTrainedModel() {
  try {
    if (fs.existsSync(TRAINED_MODEL_FILE)) {
      const data = fs.readFileSync(TRAINED_MODEL_FILE, 'utf8');
      trainedModel = JSON.parse(data);
      console.log(`✅ Loaded trained model: ${trainedModel.totalSamples} samples`);
      return true;
    }
  } catch (error) { console.error('Error loading trained model:', error.message); }
  return false;
}

function buildMarkovChainFromArray(results, order) {
  const chain = {};
  for (let i = 0; i <= results.length - order - 1; i++) {
    const state = results.slice(i, i + order).map(r => r === 'Tài' ? 'T' : 'X').join('');
    const next = results[i + order];
    if (!chain[state]) chain[state] = { 'Tài': 0, 'Xỉu': 0, total: 0 };
    chain[state][next]++;
    chain[state].total++;
  }
  return chain;
}

function predictMarkovFromModel(results, order) {
  const chain = trainedModel.markovChains[order];
  if (!chain || results.length < order) return null;
  const currentState = results.slice(0, order).map(r => r === 'Tài' ? 'T' : 'X').join('');
  const stateData = chain[currentState];
  if (!stateData || stateData.total < 3) return null;
  const taiProb = stateData['Tài'] / stateData.total;
  const xiuProb = stateData['Xỉu'] / stateData.total;
  const confidence = Math.max(taiProb, xiuProb);
  if (confidence < 0.52) return null;
  return { prediction: taiProb > xiuProb ? 'Tài' : 'Xỉu', confidence: Math.round(confidence * 100), order, state: currentState, samples: stateData.total };
}

function buildPatternLibrary(results) {
  const library = {};
  for (let len = 3; len <= 8; len++) {
    const counts = {};
    for (let i = 0; i <= results.length - len - 1; i++) {
      const pattern = results.slice(i, i + len).map(r => r === 'Tài' ? 'T' : 'X').join('');
      const next = results[i + len];
      if (!counts[pattern]) counts[pattern] = { 'Tài': 0, 'Xỉu': 0, total: 0 };
      counts[pattern][next]++;
      counts[pattern].total++;
    }
    const filtered = {};
    Object.entries(counts).forEach(([k, v]) => { if (v.total >= 20) filtered[k] = v; });
    library[`len_${len}`] = filtered;
  }
  return library;
}

function predictFromPatternLibrary(results) {
  if (!trainedModel.patternLibrary) return null;
  const candidates = [];
  for (let len = 3; len <= 8; len++) {
    const lib = trainedModel.patternLibrary[`len_${len}`];
    if (!lib || results.length < len) continue;
    const pattern = results.slice(0, len).map(r => r === 'Tài' ? 'T' : 'X').join('');
    const data = lib[pattern];
    if (!data || data.total < 10) continue;
    const taiProb = data['Tài'] / data.total, xiuProb = data['Xỉu'] / data.total;
    const confidence = Math.max(taiProb, xiuProb);
    if (confidence >= 0.55) {
      candidates.push({ prediction: taiProb > xiuProb ? 'Tài' : 'Xỉu', confidence: Math.round(confidence * 100), pattern, len, samples: data.total, priority: 5 + (len - 3) });
    }
  }
  if (candidates.length === 0) return null;
  candidates.sort((a, b) => b.confidence - a.confidence || b.samples - a.samples);
  const best = candidates[0];
  return { ...best, name: `Pattern Library (${best.pattern} → ${best.prediction}, ${best.samples} samples)`, patternId: 'markov_chain' };
}

function predictFromPatternDict(resultsArray) {
  if (resultsArray.length < 14) return null;
  const currentPattern = resultsArray.slice(0, 14).reverse().map(r => r === 'Tài' ? 'T' : 'X').join('');
  if (trainedModel.patternDict && trainedModel.patternDict[currentPattern]) {
    return { prediction: trainedModel.patternDict[currentPattern] === 'T' ? 'Tài' : 'Xỉu', confidence: 88, name: `Pattern Match (${currentPattern})`, patternId: 'pattern_dict' };
  }
  if (trainedModel.formulaMap) {
    for (const [key, rels] of Object.entries(trainedModel.formulaMap)) {
      if (currentPattern === key || currentPattern.includes(key)) {
        for (const rel of rels) {
          if (trainedModel.patternDict[rel]) {
            return { prediction: trainedModel.patternDict[rel] === 'T' ? 'Tài' : 'Xỉu', confidence: 80, name: `Formula Match (${key} -> ${rel})`, patternId: 'formula_map' };
          }
        }
      }
    }
  }
  return null;
}

// ==================== PATTERN ANALYZERS ====================
function initializePatternStats(type) {
  if (!learningData[type].patternWeights || Object.keys(learningData[type].patternWeights).length === 0) {
    learningData[type].patternWeights = { ...DEFAULT_PATTERN_WEIGHTS };
  }
  Object.keys(DEFAULT_PATTERN_WEIGHTS).forEach(pattern => {
    if (!learningData[type].patternStats[pattern]) {
      learningData[type].patternStats[pattern] = { total: 0, correct: 0, accuracy: 0.5, recentResults: [], lastAdjustment: null };
    }
  });
}

function getPatternWeight(type, patternId) {
  initializePatternStats(type);
  return learningData[type].patternWeights[patternId] || 1.0;
}

function updatePatternPerformance(type, patternId, isCorrect) {
  initializePatternStats(type);
  const stats = learningData[type].patternStats[patternId];
  if (!stats) return;
  stats.total++;
  if (isCorrect) stats.correct++;
  stats.recentResults.push(isCorrect ? 1 : 0);
  if (stats.recentResults.length > 20) stats.recentResults.shift();
  const recentAccuracy = stats.recentResults.reduce((a, b) => a + b, 0) / stats.recentResults.length;
  stats.accuracy = stats.total > 0 ? stats.correct / stats.total : 0.5;
  const oldWeight = learningData[type].patternWeights[patternId];
  let newWeight = oldWeight;
  if (stats.recentResults.length >= 5) {
    if (recentAccuracy > 0.65) newWeight = Math.min(3.0, oldWeight * 1.1);
    else if (recentAccuracy < 0.35) newWeight = Math.max(0.2, oldWeight * 0.9);
  }
  learningData[type].patternWeights[patternId] = newWeight;
  stats.lastAdjustment = new Date().toISOString();
}

function analyzeCauBet(results, type) {
  if (results.length < 3) return { detected: false };
  let streakType = results[0], streakLength = 1;
  for (let i = 1; i < results.length; i++) {
    if (results[i] === streakType) streakLength++;
    else break;
  }
  if (streakLength >= 3) {
    const weight = getPatternWeight(type, 'cau_bet');
    let shouldBreak = streakLength >= 5, confidence = 65;
    if (streakLength >= 7) { shouldBreak = true; confidence = 85; }
    else if (streakLength >= 5) { shouldBreak = true; confidence = 75; }
    else if (streakLength >= 3) { shouldBreak = false; confidence = 68; }
    return { detected: true, type: streakType, length: streakLength, prediction: shouldBreak ? (streakType === 'Tài' ? 'Xỉu' : 'Tài') : streakType, confidence: Math.round(confidence * weight), name: `Cầu Bệt ${streakLength} phiên ${streakType}`, patternId: 'cau_bet' };
  }
  return { detected: false };
}

function analyzeCauDao11(results, type) {
  if (results.length < 4) return { detected: false };
  let altLen = 1;
  for (let i = 1; i < Math.min(results.length, 10); i++) {
    if (results[i] !== results[i - 1]) altLen++;
    else break;
  }
  if (altLen >= 4) {
    const weight = getPatternWeight(type, 'cau_dao_11');
    return { detected: true, length: altLen, prediction: results[0] === 'Tài' ? 'Xỉu' : 'Tài', confidence: Math.round(Math.min(80, 65 + altLen * 2) * weight), name: `Cầu Đảo 1-1 (${altLen} phiên)`, patternId: 'cau_dao_11' };
  }
  return { detected: false };
}

function analyzeCau22(results, type) {
  if (results.length < 6) return { detected: false };
  let pairCount = 0, i = 0, pattern = [];
  while (i < results.length - 1 && pairCount < 4) {
    if (results[i] === results[i + 1]) { pattern.push(results[i]); pairCount++; i += 2; }
    else break;
  }
  if (pairCount >= 2) {
    let alt = true;
    for (let j = 1; j < pattern.length; j++) if (pattern[j] === pattern[j - 1]) { alt = false; break; }
    if (alt) {
      const last = pattern[pattern.length - 1];
      return { detected: true, pairCount, prediction: last === 'Tài' ? 'Xỉu' : 'Tài', confidence: Math.round(Math.min(78, 65 + pairCount * 3) * getPatternWeight(type, 'cau_22')), name: `Cầu 2-2 (${pairCount} cặp)`, patternId: 'cau_22' };
    }
  }
  return { detected: false };
}

function analyzeCau33(results, type) {
  if (results.length < 6) return { detected: false };
  let tc = 0, i = 0, pattern = [];
  while (i < results.length - 2) {
    if (results[i] === results[i + 1] && results[i + 1] === results[i + 2]) {
      pattern.push(results[i]); tc++; i += 3;
    } else break;
  }
  if (tc >= 1) {
    const pos = results.length % 3, last = pattern[pattern.length - 1];
    const pred = pos === 0 ? (last === 'Tài' ? 'Xỉu' : 'Tài') : last;
    return {
      detected: true, tripleCount: tc, prediction: pred,
      confidence: Math.round(Math.min(80, 68 + tc * 4) * getPatternWeight(type, 'cau_33')),
      name: `Cầu 3-3 (${tc} bộ ba)`, patternId: 'cau_33'
    };
  }
  return { detected: false };
}

function analyzeCau121(results, type) {
  if (results.length < 4) return { detected: false };
  const p = results.slice(0, 4);
  if (p[0] !== p[1] && p[1] === p[2] && p[2] !== p[3] && p[0] === p[3]) {
    return {
      detected: true, pattern: '1-2-1', prediction: p[0],
      confidence: Math.round(72 * getPatternWeight(type, 'cau_121')),
      name: 'Cầu 1-2-1', patternId: 'cau_121'
    };
  }
  return { detected: false };
}

function analyzeCau123(results, type) {
  if (results.length < 6) return { detected: false };
  const first = results[5], n2 = results.slice(3, 5), l3 = results.slice(0, 3);
  if (n2[0] === n2[1] && n2[0] !== first && l3.every(r => r === l3[0]) && l3[0] !== n2[0]) {
    return {
      detected: true, pattern: '1-2-3', prediction: first,
      confidence: Math.round(74 * getPatternWeight(type, 'cau_123')),
      name: 'Cầu 1-2-3', patternId: 'cau_123'
    };
  }
  return { detected: false };
}

function analyzeCau321(results, type) {
  if (results.length < 6) return { detected: false };
  const f3 = results.slice(3, 6), n2 = results.slice(1, 3), l1 = results[0];
  if (f3.every(r => r === f3[0]) && n2.every(r => r === n2[0]) && f3[0] !== n2[0] && l1 !== n2[0]) {
    return {
      detected: true, pattern: '3-2-1', prediction: n2[0],
      confidence: Math.round(76 * getPatternWeight(type, 'cau_321')),
      name: 'Cầu 3-2-1', patternId: 'cau_321'
    };
  }
  return { detected: false };
}

function analyzeCauNhayCoc(results, type) {
  if (results.length < 6) return { detected: false };
  const skip = [];
  for (let i = 0; i < Math.min(results.length, 12); i += 2) skip.push(results[i]);
  if (skip.length >= 3) {
    const w = getPatternWeight(type, 'cau_nhay_coc');
    if (skip.slice(0, 3).every(r => r === skip[0])) {
      return {
        detected: true, prediction: skip[0], confidence: Math.round(68 * w),
        name: 'Cầu Nhảy Cóc', patternId: 'cau_nhay_coc'
      };
    }
    let alt = true;
    for (let i = 1; i < skip.length - 1; i++) if (skip[i] === skip[i - 1]) { alt = false; break; }
    if (alt && skip.length >= 3) {
      return {
        detected: true, prediction: skip[0] === 'Tài' ? 'Xỉu' : 'Tài',
        confidence: Math.round(66 * w), name: 'Cầu Nhảy Cóc Đảo', patternId: 'cau_nhay_coc'
      };
    }
  }
  return { detected: false };
}

function analyzeCauNhipNghieng(results, type) {
  if (results.length < 5) return { detected: false };
  const last5 = results.slice(0, 5);
  const t5 = last5.filter(r => r === 'Tài').length;
  const w = getPatternWeight(type, 'cau_nhip_nghieng');
  if (t5 >= 4) return {
    detected: true, prediction: 'Tài', confidence: Math.round(70 * w),
    name: `Cầu Nhịp Nghiêng (${t5}/5 Tài)`, patternId: 'cau_nhip_nghieng'
  };
  if (t5 <= 1) return {
    detected: true, prediction: 'Xỉu', confidence: Math.round(70 * w),
    name: `Cầu Nhịp Nghiêng (${5-t5}/5 Xỉu)`, patternId: 'cau_nhip_nghieng'
  };
  return { detected: false };
}

function analyzeCau3Van1(results, type) {
  if (results.length < 4) return { detected: false };
  const t = results.slice(0, 4).filter(r => r === 'Tài').length;
  const w = getPatternWeight(type, 'cau_3van1');
  if (t === 3) return {
    detected: true, prediction: 'Xỉu', confidence: Math.round(68 * w),
    name: 'Cầu 3 Ván 1 (3T-1X) → Xỉu', patternId: 'cau_3van1'
  };
  if (t === 1) return {
    detected: true, prediction: 'Tài', confidence: Math.round(68 * w),
    name: 'Cầu 3 Ván 1 (3X-1T) → Tài', patternId: 'cau_3van1'
  };
  return { detected: false };
}

function analyzeCauBeCau(results, type) {
  if (results.length < 8) return { detected: false };
  const rs = analyzeCauBet(results, type);
  if (rs.detected && rs.length >= 4) {
    const before = results.slice(rs.length, rs.length + 4);
    const pp = analyzeCauBet(before, type);
    if (pp.detected && pp.type !== rs.type) {
      return {
        detected: true, prediction: rs.type === 'Tài' ? 'Xỉu' : 'Tài',
        confidence: Math.round(76 * getPatternWeight(type, 'cau_be_cau')),
        name: 'Cầu Bẻ Cầu', patternId: 'cau_be_cau'
      };
    }
  }
  return { detected: false };
}

function analyzeCauTuNhien(results, type) {
  if (results.length < 2) return { detected: false };
  return {
    detected: true, prediction: results[0],
    confidence: Math.round(60 * getPatternWeight(type, 'cau_tu_nhien')),
    name: 'Cầu Tự Nhiên (Theo Ván Trước)', patternId: 'cau_tu_nhien'
  };
}

function analyzeCauRong(results, type) {
  if (results.length < 6) return { detected: false };
  let sl = 1;
  for (let i = 1; i < results.length; i++) if (results[i] === results[0]) sl++; else break;
  if (sl >= 6) return {
    detected: true, prediction: results[0] === 'Tài' ? 'Xỉu' : 'Tài',
    confidence: Math.round(Math.min(88, 75 + sl) * getPatternWeight(type, 'cau_rong')),
    name: `Cầu Rồng ${sl} phiên (Bẻ mạnh)`, patternId: 'cau_rong'
  };
  return { detected: false };
}

function analyzeSmartBet(results, type) {
  if (results.length < 10) return { detected: false };
  const l5 = results.slice(0, 5), p5 = results.slice(5, 10);
  const tl5 = l5.filter(r => r === 'Tài').length, tp5 = p5.filter(r => r === 'Tài').length;
  const w = getPatternWeight(type, 'smart_bet');
  if ((tl5 >= 4 && tp5 <= 1) || (tl5 <= 1 && tp5 >= 4)) {
    const cd = tl5 >= 4 ? 'Tài' : 'Xỉu';
    return {
      detected: true, prediction: cd === 'Tài' ? 'Xỉu' : 'Tài',
      confidence: Math.round(78 * w),
      name: `Đảo Xu Hướng (${tl5}T-${5-tl5}X → ${tp5}T-${5-tp5}X)`, patternId: 'smart_bet'
    };
  }
  return { detected: false };
}

function analyzeBreakStreak(results, type) {
  if (results.length < 5) return { detected: false };
  let st = results[0], sl = 1;
  for (let i = 1; i < results.length; i++) if (results[i] === st) sl++; else break;
  if (sl >= 5) {
    const pred = st === 'Tài' ? 'Xỉu' : 'Tài';
    return {
      detected: true, prediction: pred,
      confidence: Math.round(Math.min(85, 70 + sl) * getPatternWeight(type, 'break_streak')),
      name: `Bẻ Chuỗi ${sl} (${st} → ${pred})`, patternId: 'break_streak'
    };
  }
  return { detected: false };
}

function analyzeAlternatingBreak(results, type) {
  if (results.length < 6) return { detected: false };
  let ac = 0;
  for (let i = 0; i < results.length - 1; i++) {
    if (results[i] !== results[i + 1]) ac++; else break;
  }
  if (ac >= 6) {
    const pred = results[0] === 'Tài' ? 'Xỉu' : 'Tài';
    return {
      detected: true, prediction: pred,
      confidence: Math.round(Math.min(82, 68 + ac) * getPatternWeight(type, 'alternating_break')),
      name: `Bẻ Đảo ${ac} phiên → ${pred}`, patternId: 'alternating_break'
    };
  }
  return { detected: false };
}

function analyzeDoublePairBreak(results, type) {
  if (results.length < 8) return { detected: false };
  const p1 = results[0] === results[1], p2 = results[2] === results[3];
  const p3 = results[4] === results[5], p4 = results[6] === results[7];
  if (p1 && p2 && p3 && p4) {
    const pt1 = results[0], pt2 = results[2];
    const same = pt1 === pt2 && pt2 === results[4] && results[4] === results[6];
    const w = getPatternWeight(type, 'double_pair_break');
    if (same) {
      const pred = pt1 === 'Tài' ? 'Xỉu' : 'Tài';
      return {
        detected: true, prediction: pred, confidence: Math.round(84 * w),
        name: `4 Cặp Cùng ${pt1} → Bẻ ${pred}`, patternId: 'double_pair_break'
      };
    }
    const alt = pt1 !== pt2 && pt2 !== results[4] && results[4] !== results[6];
    if (alt) {
      const pred = results[0] === 'Tài' ? 'Xỉu' : 'Tài';
      return {
        detected: true, prediction: pred, confidence: Math.round(78 * w),
        name: `Cặp Đảo Xen Kẽ → Bẻ ${pred}`, patternId: 'double_pair_break'
      };
    }
  }
  return { detected: false };
}

function analyzeTriplePattern(results, type) {
  if (results.length < 9) return { detected: false };
  const t1 = results[0] === results[1] && results[1] === results[2];
  const t2 = results[3] === results[4] && results[4] === results[5];
  const t3 = results[6] === results[7] && results[7] === results[8];
  if (t1 && t2 && t3) {
    const tt1 = results[0], tt2 = results[3], tt3 = results[6];
    const w = getPatternWeight(type, 'triple_pattern');
    if (tt1 === tt2 && tt2 === tt3) {
      const pred = tt1 === 'Tài' ? 'Xỉu' : 'Tài';
      return {
        detected: true, prediction: pred, confidence: Math.round(88 * w),
        name: `3 Bộ Ba Cùng ${tt1} → Bẻ ${pred}`, patternId: 'triple_pattern'
      };
    }
  }
  return { detected: false };
}

function analyzeTongPhanTich(data, type) {
  if (data.length < 10) return { detected: false };
  const r10 = data.slice(0, 10);
  const sums = r10.map(d => d.Tong);
  const results = r10.map(d => d.Ket_qua);
  const s5a = sums.slice(5, 10).reduce((a, b) => a + b, 0) / 5;
  const s5b = sums.slice(0, 5).reduce((a, b) => a + b, 0) / 5;
  const trend = s5b - s5a;
  if (trend > 1.5) return {
    detected: true, prediction: 'Xỉu', confidence: Math.round(75 + Math.abs(trend) * 3),
    name: `Tổng tăng ${trend.toFixed(1)} → Xỉu`, patternId: 'tong_phan_tich'
  };
  if (trend < -1.5) return {
    detected: true, prediction: 'Tài', confidence: Math.round(75 + Math.abs(trend) * 3),
    name: `Tổng giảm ${Math.abs(trend).toFixed(1)} → Tài`, patternId: 'tong_phan_tich'
  };
  const tc = results.filter(r => r === 'Tài').length, xc = results.filter(r => r === 'Xỉu').length;
  if (Math.abs(tc - xc) >= 3) {
    const lech = tc > xc ? 'Tài' : 'Xỉu';
    const pred = lech === 'Tài' ? 'Xỉu' : 'Tài';
    return {
      detected: true, prediction: pred,
      confidence: Math.round(70 + Math.abs(tc - xc) * 3),
      name: `Lệch ${Math.abs(tc-xc)} về ${lech} → ${pred}`, patternId: 'tong_phan_tich'
    };
  }
  return { detected: false };
}

function analyzeXuHuongManh(results, type) {
  if (results.length < 8) return { detected: false };
  const r8 = results.slice(0, 8);
  const tc = r8.filter(r => r === 'Tài').length;
  if (tc >= 6) return {
    detected: true, prediction: 'Xỉu', confidence: Math.round(80 + tc * 2),
    name: `Xu Hướng Mạnh (${tc}/8 Tài → Đảo Xỉu)`, patternId: 'xu_huong_manh'
  };
  if (tc <= 2) return {
    detected: true, prediction: 'Tài', confidence: Math.round(80 + (8 - tc) * 2),
    name: `Xu Hướng Mạnh (${8-tc}/8 Xỉu → Đảo Tài)`, patternId: 'xu_huong_manh'
  };
  return { detected: false };
}

function analyzeDaoChieu(results, type) {
  if (results.length < 5) return { detected: false };
  const r5 = results.slice(0, 5);
  let alt = true;
  for (let i = 0; i < r5.length - 1; i++) if (r5[i] === r5[i + 1]) { alt = false; break; }
  if (alt) {
    const pred = r5[0] === 'Tài' ? 'Xỉu' : 'Tài';
    return {
      detected: true, prediction: pred, confidence: 75,
      name: `Đảo Chiều (${r5.join('-')} → ${pred})`, patternId: 'dao_chieu'
    };
  }
  return { detected: false };
}

function analyzeDistribution(data, type, windowSize = 50) {
  const win = data.slice(0, windowSize);
  const tc = win.filter(d => d.Ket_qua === 'Tài').length;
  const xc = win.length - tc;
  return {
    taiPercent: (tc / win.length) * 100, xiuPercent: (xc / win.length) * 100,
    taiCount: tc, xiuCount: xc, total: win.length,
    imbalance: Math.abs(tc - xc) / win.length
  };
}

// ==================== MAIN PREDICTION (ĐÃ FIX LOGIC) ====================
function calculateAdvancedPrediction(data, type) {
  const last50 = data.slice(0, 50);
  const results = last50.map(d => d.Ket_qua);
  initializePatternStats(type);
  
  let predictions = [], factors = [];
  
  // 1. ƯU TIÊN SỐ 1: PATTERN DICT (10K CẦU) - Nếu khớp thì tin tưởng tuyệt đối
  const pd = predictFromPatternDict(results);
  if (pd) {
    return {
      prediction: pd.prediction,
      confidence: 85, // Cố định 85% nếu khớp 10k cầu
      factors: [`[10K CẦU] ${pd.name}`],
      allPatterns: [pd],
      detailedAnalysis: { totalPatterns: 1, topPattern: pd.name, trained: trainedModel.trained }
    };
  }

  // 2. ƯU TIÊN SỐ 2: MARKOV CHAINS TỪ FILE 17K (Chỉ xét bậc cao 5-9)
  if (trainedModel.trained) {
    for (let order = 9; order >= 5; order--) { 
      const p = predictMarkovFromModel(results, order);
      if (p && p.confidence >= 70) {
        return {
          prediction: p.prediction,
          confidence: p.confidence, 
          factors: [`[17K MARKOV BẬC ${order}] ${p.state} -> ${p.prediction}`],
          allPatterns: [{ ...p, patternId: 'markov_chain' }],
          detailedAnalysis: { totalPatterns: 1, topPattern: `Markov bậc ${order}`, trained: true }
        };
      }
    }
  }
  
  // 3. NẾU KHÔNG KHỚP CẦU NÀO RÕ RÀNG, DÙNG CÁC PATTERN CƠ BẢN ĐỂ CHẤM ĐIỂM (Giảm trọng số)
  const analyzers = [
    { fn: analyzeTongPhanTich(last50, type), priority: 5 },
    { fn: analyzeXuHuongManh(results, type), priority: 4 },
    { fn: analyzeCauRong(results, type), priority: 4 }, 
    { fn: analyzeCauBet(results, type), priority: 3 },
    { fn: analyzeCauDao11(results, type), priority: 3 },
    { fn: analyzeCau22(results, type), priority: 2 },
    { fn: analyzeSmartBet(results, type), priority: 2 }
  ];
  
  let taiScore = 0, xiuScore = 0;
  
  analyzers.forEach(item => {
    if (item.fn && item.fn.detected) {
      const score = (item.fn.confidence / 100) * item.priority;
      if (item.fn.prediction === 'Tài') taiScore += score;
      else xiuScore += score;
      
      predictions.push({ 
        prediction: item.fn.prediction, 
        confidence: item.fn.confidence, 
        name: item.fn.name 
      });
      factors.push(item.fn.name);
    }
  });
  
  // Nếu không có tín hiệu nào, theo số đông 10 phiên gần nhất
  if (predictions.length === 0) {
    const tc = results.slice(0, 10).filter(r => r === 'Tài').length;
    const pred = tc >= 5 ? 'Tài' : 'Xỉu';
    return {
      prediction: pred,
      confidence: 55,
      factors: ['Không có cầu rõ ràng - Đánh theo số đông 10 phiên'],
      allPatterns: [],
      detailedAnalysis: { totalPatterns: 0, topPattern: 'N/A', trained: trainedModel.trained }
    };
  }
  
  let finalPrediction = taiScore >= xiuScore ? 'Tài' : 'Xỉu';
  
  // Tính độ tự tin thực tế (Không thổi phồng)
  let totalScore = taiScore + xiuScore;
  let winScore = finalPrediction === 'Tài' ? taiScore : xiuScore;
  let winRate = winScore / totalScore; 
  
  // Công thức tính Conf: Khóa trần ở mức 78% để phản ánh đúng thực tế
  let finalConf = Math.round(55 + (winRate * 20)); 
  finalConf = Math.max(55, Math.min(78, finalConf)); 
  
  return {
    prediction: finalPrediction,
    confidence: finalConf,
    factors: factors.length > 0 ? factors : ['Phân tích tổng hợp'],
    allPatterns: predictions,
    detailedAnalysis: {
      totalPatterns: predictions.length,
      taiScore, xiuScore,
      topPattern: predictions[0]?.name || 'N/A',
      trained: trainedModel.trained
    }
  };
}

function getAdaptiveConfidenceBoost(type) {
  const ra = learningData[type].recentAccuracy;
  if (ra.length < 10) return 0;
  const acc = ra.reduce((a, b) => a + b, 0) / ra.length;
  if (acc > 0.70) return 10;
  if (acc > 0.60) return 6;
  if (acc > 0.50) return 3;
  if (acc < 0.30) return -10;
  if (acc < 0.40) return -6;
  return 0;
}

// ==================== API DATA FETCH ====================
function transformApiData(apiData) {
  if (!apiData || !apiData.list || !Array.isArray(apiData.list)) return null;
  return apiData.list.map(item => ({
    Phien: item.id,
    Ket_qua: item.resultTruyenThong === 'TAI' ? 'Tài' : 'Xỉu',
    Xuc_xac_1: item.dices[0], Xuc_xac_2: item.dices[1], Xuc_xac_3: item.dices[2],
    Tong: item.point
  }));
}

async function fetchDataHu() {
  try { return transformApiData((await axios.get(API_URL_HU, { timeout: 10000 })).data); }
  catch (e) { console.error('Fetch HU:', e.message); return null; }
}

async function fetchDataMd5() {
  try { return transformApiData((await axios.get(API_URL_MD5, { timeout: 10000 })).data); }
  catch (e) { console.error('Fetch MD5:', e.message); return null; }
}

// ==================== PATTERN_10 ====================
function updatePattern10(type, data) {
  const last10 = data.slice(0, 10);
  lastPattern10[type] = last10.map(d => d.Ket_qua === 'Tài' ? 'T' : 'X');
  learningData[type].pattern10History = lastPattern10[type];
  return lastPattern10[type].join('');
}

// ==================== DUDOAN HISTORY (ĐÃ FIX LOGIC ĐỒNG BỘ) ====================
function saveDudoanRecord(type, phienTruoc, resultTruoc, phienNay, duDoan, doTinCay) {
  const rec = {
    phien_truoc: phienTruoc, 
    result: resultTruoc, // Kết quả của phiên vừa xổ
    Phien_nay: phienNay, 
    Du_doan: duDoan, // Dự đoán cho phiên NÀY
    Do_tin_cay: doTinCay, 
    Trang_thai: '', // Trạng thái này sẽ được điền khi phiên này có kết quả
    timestamp: new Date().toISOString()
  };
  
  dudoanHistory[type].unshift(rec);
  if (dudoanHistory[type].length > MAX_DUDOAN_HISTORY) {
    dudoanHistory[type].pop();
  }
  return rec;
}

// ==================== PREDICTION RECORD ====================
function recordPrediction(type, phien, prediction, confidence, patterns) {
  const rec = {
    phien: phien.toString(), prediction, confidence, patterns,
    timestamp: new Date().toISOString(),
    verified: false, actual: null, isCorrect: null
  };
  learningData[type].predictions.unshift(rec);
  learningData[type].totalPredictions++;
  if (learningData[type].predictions.length > 500) {
    learningData[type].predictions = learningData[type].predictions.slice(0, 500);
  }
  saveLearningData();
}

async function verifyPredictions(type, currentData) {
  let updated = false;
  for (const pred of learningData[type].predictions) {
    if (pred.verified) continue;
    const actual = currentData.find(d => d.Phien.toString() === pred.phien);
    if (actual) {
      pred.verified = true;
      pred.actual = actual.Ket_qua;
      pred.isCorrect = pred.actual === pred.prediction;
      if (pred.isCorrect) {
        learningData[type].correctPredictions++;
        learningData[type].streakAnalysis.wins++;
        learningData[type].streakAnalysis.currentStreak = 
          learningData[type].streakAnalysis.currentStreak >= 0 
            ? learningData[type].streakAnalysis.currentStreak + 1 : 1;
        if (learningData[type].streakAnalysis.currentStreak > learningData[type].streakAnalysis.bestStreak)
          learningData[type].streakAnalysis.bestStreak = learningData[type].streakAnalysis.currentStreak;
      } else {
        learningData[type].streakAnalysis.losses++;
        learningData[type].streakAnalysis.currentStreak = 
          learningData[type].streakAnalysis.currentStreak <= 0 
            ? learningData[type].streakAnalysis.currentStreak - 1 : -1;
        if (learningData[type].streakAnalysis.currentStreak < learningData[type].streakAnalysis.worstStreak)
          learningData[type].streakAnalysis.worstStreak = learningData[type].streakAnalysis.currentStreak;
      }
      learningData[type].recentAccuracy.push(pred.isCorrect ? 1 : 0);
      if (learningData[type].recentAccuracy.length > 50) learningData[type].recentAccuracy.shift();
      if (pred.patterns) {
        pred.patterns.forEach(pn => {
          const pid = getPatternIdFromName(pn);
          if (pid) updatePatternPerformance(type, pid, pred.isCorrect);
        });
      }
      updated = true;
    }
  }
  if (updated) {
    learningData[type].lastUpdate = new Date().toISOString();
    saveLearningData();
  }
}

function getPatternIdFromName(name) {
  const m = {
    'Cầu Bệt': 'cau_bet', 'Cầu Đảo 1-1': 'cau_dao_11',
    'Cầu 2-2': 'cau_22', 'Cầu 3-3': 'cau_33',
    'Cầu 1-2-1': 'cau_121', 'Cầu 1-2-3': 'cau_123', 'Cầu 3-2-1': 'cau_321',
    'Cầu Nhảy Cóc': 'cau_nhay_coc', 'Cầu Nhịp Nghiêng': 'cau_nhip_nghieng',
    'Cầu 3 Ván 1': 'cau_3van1', 'Cầu Bẻ Cầu': 'cau_be_cau',
    'Cầu Rồng': 'cau_rong', 'Smart Bet': 'smart_bet',
    'Tổng tăng': 'tong_phan_tich', 'Tổng giảm': 'tong_phan_tich', 'Lệch': 'tong_phan_tich',
    'Xu Hướng Mạnh': 'xu_huong_manh', 'Đảo Chiều': 'dao_chieu',
    'Bẻ Chuỗi': 'break_streak', 'Bẻ Đảo': 'alternating_break',
    '3 Bộ Ba': 'triple_pattern', 'Cặp Đảo': 'double_pair_break',
    'Markov': 'markov_chain', 'Pattern Library': 'markov_chain',
    'Frequency': 'frequency_analysis', 'Cầu Tự Nhiên': 'cau_tu_nhien',
    'Pattern Match': 'pattern_dict', 'Formula Match': 'formula_map'
  };
  for (const [k, v] of Object.entries(m)) if (name.includes(k)) return v;
  return null;
}

async function updateHistoryStatus(type) {
  try {
    const data = type === 'hu' ? await fetchDataHu() : await fetchDataMd5();
    if (!data || data.length === 0) return;
    let updated = false;
    for (const rec of predictionHistory[type]) {
      if (rec.ket_qua_du_doan && rec.ket_qua_du_doan !== '') continue;
      const actual = data.find(d => d.Phien.toString() === rec.Phien_hien_tai);
      if (actual) {
        rec.ket_qua_du_doan = rec.Du_doan === actual.Ket_qua ? 'Đúng ✅' : 'Sai ❌';
        updated = true;
      }
    }
    if (updated) savePredictionHistory();
  } catch (e) { console.error(`Update ${type}:`, e.message); }
}

function savePredictionToHistory(type, phien, prediction, confidence, latestData) {
  const rec = {
    Phien: latestData.Phien,
    Xuc_xac_1: latestData.Xuc_xac_1, Xuc_xac_2: latestData.Xuc_xac_2, Xuc_xac_3: latestData.Xuc_xac_3,
    Tong: latestData.Tong, Ket_qua: latestData.Ket_qua,
    Do_tin_cay: `${confidence}%`,
    Phien_hien_tai: phien.toString(),
    Du_doan: prediction, ket_qua_du_doan: '',
    id: AUTHOR_ID, timestamp: new Date().toISOString()
  };
  predictionHistory[type].unshift(rec);
  if (predictionHistory[type].length > MAX_HISTORY)
    predictionHistory[type] = predictionHistory[type].slice(0, MAX_HISTORY);
  return rec;
}

async function autoProcessPredictions() {
  try {
    const dataHu = await fetchDataHu();
    if (dataHu && dataHu.length > 0) {
      const nextPhien = dataHu[0].Phien + 1;
      if (lastProcessedPhien.hu !== nextPhien) {
        
        // Cập nhật trạng thái cho dự đoán của phiên TRƯỚC ĐÓ (Nếu có)
        if (dudoanHistory.hu.length > 0) {
           const lastRecord = dudoanHistory.hu[0];
           if (lastRecord.Phien_nay === dataHu[0].Phien.toString() && lastRecord.Trang_thai === '') {
               lastRecord.Trang_thai = lastRecord.Du_doan === dataHu[0].Ket_qua ? 'AUDU LẠI BÚ✅' : 'DITMEMAY LẠI GÃY❌';
           }
        }

        await verifyPredictions('hu', dataHu);
        updatePattern10('hu', dataHu);
        const result = calculateAdvancedPrediction(dataHu, 'hu');
        savePredictionToHistory('hu', nextPhien, result.prediction, result.confidence, dataHu[0]);
        recordPrediction('hu', nextPhien, result.prediction, result.confidence, result.factors);
        saveDudoanRecord('hu', dataHu[0].Phien, dataHu[0].Ket_qua, nextPhien, result.prediction, `${result.confidence}%`);
        lastProcessedPhien.hu = nextPhien;
        console.log(`[Auto] HU ${nextPhien}: ${result.prediction} (${result.confidence}%)`);
      }
    }
    const dataMd5 = await fetchDataMd5();
    if (dataMd5 && dataMd5.length > 0) {
      const nextPhien = dataMd5[0].Phien + 1;
      if (lastProcessedPhien.md5 !== nextPhien) {
        
        // Cập nhật trạng thái cho dự đoán của phiên TRƯỚC ĐÓ (Nếu có)
        if (dudoanHistory.md5.length > 0) {
           const lastRecord = dudoanHistory.md5[0];
           if (lastRecord.Phien_nay === dataMd5[0].Phien.toString() && lastRecord.Trang_thai === '') {
               lastRecord.Trang_thai = lastRecord.Du_doan === dataMd5[0].Ket_qua ? 'AUDU LẠI BÚ✅' : 'DITMEMAY LẠI GÃY❌';
           }
        }

        await verifyPredictions('md5', dataMd5);
        updatePattern10('md5', dataMd5);
        const result = calculateAdvancedPrediction(dataMd5, 'md5');
        savePredictionToHistory('md5', nextPhien, result.prediction, result.confidence, dataMd5[0]);
        recordPrediction('md5', nextPhien, result.prediction, result.confidence, result.factors);
        saveDudoanRecord('md5', dataMd5[0].Phien, dataMd5[0].Ket_qua, nextPhien, result.prediction, `${result.confidence}%`);
        lastProcessedPhien.md5 = nextPhien;
        console.log(`[Auto] MD5 ${nextPhien}: ${result.prediction} (${result.confidence}%)`);
      }
    }
    await updateHistoryStatus('hu');
    await updateHistoryStatus('md5');
    savePredictionHistory();
    saveLearningData();
  } catch (e) { console.error('[Auto]:', e.message); }
}

function startAutoSaveTask() {
  console.log(`⏰ Auto-save task (every ${AUTO_SAVE_INTERVAL/1000}s)`);
  setTimeout(() => autoProcessPredictions(), 5000);
  setInterval(() => autoProcessPredictions(), AUTO_SAVE_INTERVAL);
}

// ==================== ENDPOINTS ====================
app.get('/', (req, res) => {
  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.send(`🔥 Tài Xỉu VIP PRO ULTRA MAX 6.0 - 17K + 10K CAU (BẢN FIX)
📡 Endpoints:
  GET /api/tx              - Dự đoán Hũ (pattern_10)
  GET /api/tx/md5          - Dự đoán MD5 (pattern_10)
  GET /api/history/dudoan  - 1000 phiên đúng/sai
  GET /api/history         - Lịch sử chi tiết
  GET /api/analysis        - Phân tích chi tiết
  GET /api/learning        - Stats AI
  GET /api/pattern10       - 10 cầu gần nhất
  GET /api/training-status - Trạng thái học
  GET /retrain             - Force retrain
  GET /reset-learning      - Reset learning

🧠 Trained: ${trainedModel.trained ? '✅' : '❌'}
📚 Samples: ${trainedModel.totalSamples}
🎯 Patterns: ${Object.keys(trainedModel.patternDict || {}).length}
🔗 Formulas: ${Object.keys(trainedModel.formulaMap || {}).length}
👤 ${AUTHOR_ID}`);
});

app.get('/api/tx', async (req, res) => {
  try {
    const data = await fetchDataHu();
    if (!data || data.length === 0) return res.status(500).json({ error: 'Không lấy được dữ liệu' });
    
    // Cập nhật trạng thái cho dự đoán của phiên TRƯỚC ĐÓ (Nếu có)
    if (dudoanHistory.hu.length > 0) {
       const lastRecord = dudoanHistory.hu[0];
       if (lastRecord.Phien_nay === data[0].Phien.toString() && lastRecord.Trang_thai === '') {
           lastRecord.Trang_thai = lastRecord.Du_doan === data[0].Ket_qua ? 'AUDU LẠI BÚ✅' : 'DITMEMAY LẠI GÃY❌';
       }
    }

    await verifyPredictions('hu', data);
    updatePattern10('hu', data);
    const nextPhien = data[0].Phien + 1;
    const result = calculateAdvancedPrediction(data, 'hu');
    
    const rec = savePredictionToHistory('hu', nextPhien, result.prediction, result.confidence, data[0]);
    recordPrediction('hu', nextPhien, result.prediction, result.confidence, result.factors);
    saveDudoanRecord('hu', data[0].Phien, data[0].Ket_qua, nextPhien, result.prediction, `${result.confidence}%`);
    setTimeout(async () => await updateHistoryStatus('hu'), 5000);
    
    res.json({
      Phien: rec.Phien,
      Xuc_xac_1: rec.Xuc_xac_1, Xuc_xac_2: rec.Xuc_xac_2, Xuc_xac_3: rec.Xuc_xac_3,
      Tong: rec.Tong, Ket_qua: rec.Ket_qua,
      Do_tin_cay: rec.Do_tin_cay,
      Phien_hien_tai: rec.Phien_hien_tai,
      Du_doan: rec.Du_doan,
      pattern_10: lastPattern10.hu.join(''),
      ket_qua_du_doan: rec.ket_qua_du_doan || '',
      id: rec.id,
      trained_on: trainedModel.trained ? `${trainedModel.totalSamples} samples` : 'not trained',
      Phan_tich: result.factors
    });
  } catch (e) { console.error(e); res.status(500).json({ error: 'Lỗi server' }); }
});

app.get('/api/tx/md5', async (req, res) => {
  try {
    const data = await fetchDataMd5();
    if (!data || data.length === 0) return res.status(500).json({ error: 'Không lấy được dữ liệu' });
    
    // Cập nhật trạng thái cho dự đoán của phiên TRƯỚC ĐÓ (Nếu có)
    if (dudoanHistory.md5.length > 0) {
       const lastRecord = dudoanHistory.md5[0];
       if (lastRecord.Phien_nay === data[0].Phien.toString() && lastRecord.Trang_thai === '') {
           lastRecord.Trang_thai = lastRecord.Du_doan === data[0].Ket_qua ? 'AUDU LẠI BÚ✅' : 'DITMEMAY LẠI GÃY❌';
       }
    }

    await verifyPredictions('md5', data);
    updatePattern10('md5', data);
    const nextPhien = data[0].Phien + 1;
    const result = calculateAdvancedPrediction(data, 'md5');
    
    const rec = savePredictionToHistory('md5', nextPhien, result.prediction, result.confidence, data[0]);
    recordPrediction('md5', nextPhien, result.prediction, result.confidence, result.factors);
    saveDudoanRecord('md5', data[0].Phien, data[0].Ket_qua, nextPhien, result.prediction, `${result.confidence}%`);
    setTimeout(async () => await updateHistoryStatus('md5'), 5000);
    
    res.json({
      Phien: rec.Phien,
      Xuc_xac_1: rec.Xuc_xac_1, Xuc_xac_2: rec.Xuc_xac_2, Xuc_xac_3: rec.Xuc_xac_3,
      Tong: rec.Tong, Ket_qua: rec.Ket_qua,
      Do_tin_cay: rec.Do_tin_cay,
      Phien_hien_tai: rec.Phien_hien_tai,
      Du_doan: rec.Du_doan,
      pattern_10: lastPattern10.md5.join(''),
      ket_qua_du_doan: rec.ket_qua_du_doan || '',
      id: rec.id,
      trained_on: trainedModel.trained ? `${trainedModel.totalSamples} samples` : 'not trained',
      Phan_tich: result.factors
    });
  } catch (e) { console.error(e); res.status(500).json({ error: 'Lỗi server' }); }
});

app.get('/api/history/dudoan', async (req, res) => {
  try {
    const type = req.query.type || 'hu';
    if (!dudoanHistory[type]) dudoanHistory[type] = [];
    
    // Tự động quét và cập nhật lại trạng thái ĐÚNG/SAI cho toàn bộ lịch sử
    for (let i = 0; i < dudoanHistory[type].length - 1; i++) {
        let current = dudoanHistory[type][i];
        let next = dudoanHistory[type][i + 1];
        if (current.phien_truoc === next.Phien_nay) {
            current.Trang_thai = current.Du_doan === current.result ? 'AUDU LẠI BÚ✅' : 'DITMEMAY LẠI GÃY❌';
        }
    }
    
    res.json({
      type: type === 'hu' ? 'Tài Xỉu Hũ' : 'Tài Xỉu MD5',
      total: dudoanHistory[type].length,
      history: dudoanHistory[type].slice(0, 1000)
    });
  } catch (e) { res.status(500).json({ error: 'Lỗi server' }); }
});

app.get('/api/history', async (req, res) => {
  try {
    const type = req.query.type || 'hu';
    await updateHistoryStatus(type);
    res.json({
      type: type === 'hu' ? 'Tài Xỉu Hũ' : 'Tài Xỉu MD5',
      history: predictionHistory[type],
      total: predictionHistory[type].length
    });
  } catch (e) {
    res.json({ history: predictionHistory[req.query.type || 'hu'] || [] });
  }
});

app.get('/api/analysis', async (req, res) => {
  try {
    const type = req.query.type || 'hu';
    const data = type === 'hu' ? await fetchDataHu() : await fetchDataMd5();
    if (!data || data.length === 0) return res.status(500).json({ error: 'Không lấy được dữ liệu' });
    await verifyPredictions(type, data);
    const result = calculateAdvancedPrediction(data, type);
    res.json({
      prediction: result.prediction, confidence: result.confidence,
      factors: result.factors, analysis: result.detailedAnalysis,
      pattern_10: lastPattern10[type].join('')
    });
  } catch (e) { res.status(500).json({ error: 'Lỗi server' }); }
});

app.get('/api/learning', (req, res) => {
  const type = req.query.type || 'hu';
  const stats = learningData[type];
  const acc = stats.totalPredictions > 0
    ? (stats.correctPredictions / stats.totalPredictions * 100).toFixed(2) : 0;
  res.json({
    type: type === 'hu' ? 'Tài Xỉu Hũ' : 'Tài Xỉu MD5',
    trainedModel: {
      trained: trainedModel.trained,
      totalSamples: trainedModel.totalSamples,
      trainStartTime: trainedModel.trainStartTime,
      trainEndTime: trainedModel.trainEndTime,
      taiPercent: trainedModel.taiPercent?.toFixed(2) + '%',
      xiuPercent: trainedModel.xiuPercent?.toFixed(2) + '%',
      markovOrders: Object.keys(trainedModel.markovChains || {}).length,
      patternsCount: Object.keys(trainedModel.patternDict || {}).length,
      formulasCount: Object.keys(trainedModel.formulaMap || {}).length
    },
    livePredictions: {
      total: stats.totalPredictions,
      correct: stats.correctPredictions,
      accuracy: `${acc}%`,
      streak: stats.streakAnalysis
    },
    patternWeights: stats.patternWeights
  });
});

app.get('/api/pattern10', async (req, res) => {
  try {
    const type = req.query.type || 'hu';
    const data = type === 'hu' ? await fetchDataHu() : await fetchDataMd5();
    if (!data || data.length === 0) return res.status(500).json({ error: 'Không lấy được dữ liệu' });
    updatePattern10(type, data);
    res.json({
      type: type === 'hu' ? 'Tài Xỉu Hũ' : 'Tài Xỉu MD5',
      pattern_10: lastPattern10[type].join(''),
      details: data.slice(0, 10).map(d => ({ Phien: d.Phien, Ket_qua: d.Ket_qua, Tong: d.Tong }))
    });
  } catch (e) { res.status(500).json({ error: 'Lỗi server' }); }
});

app.get('/api/training-status', (req, res) => {
  res.json({
    trained: trainedModel.trained,
    totalSamples: trainedModel.totalSamples,
    trainStartTime: trainedModel.trainStartTime,
    trainEndTime: trainedModel.trainEndTime,
    duration: trainedModel.trainStartTime && trainedModel.trainEndTime
      ? `${new Date(trainedModel.trainEndTime) - new Date(trainedModel.trainStartTime)}ms`
      : null,
    markovOrders: Object.keys(trainedModel.markovChains || {}).length,
    markovStates: trainedModel.markovChains
      ? Object.fromEntries(Object.entries(trainedModel.markovChains).map(([k, v]) => [k, Object.keys(v).length]))
      : {},
    frequencyData: trainedModel.frequencyData,
    transitions: trainedModel.transitions,
    patternDictSize: Object.keys(trainedModel.patternDict || {}).length,
    formulaMapSize: Object.keys(trainedModel.formulaMap || {}).length,
    taiPercent: trainedModel.taiPercent,
    xiuPercent: trainedModel.xiuPercent
  });
});

app.get('/retrain', async (req, res) => {
  console.log('🔄 Force retrain từ CSV...');
  const success = await loadAndTrainFromCSV();
  if (success) {
    await loadPatternCau();
    await loadFormulaCau();
    fs.writeFileSync(TRAINED_MODEL_FILE, JSON.stringify(trainedModel));
  }
  res.json({ success, totalSamples: trainedModel.totalSamples, patterns: Object.keys(trainedModel.patternDict || {}).length });
});

app.get('/reset-learning', (req, res) => {
  learningData = { hu: createEmptyLearningData(), md5: createEmptyLearningData() };
  saveLearningData();
  res.json({ message: 'Reset OK' });
});

// ==================== START ====================
loadLearningData();
loadPredictionHistory();

if (loadTrainedModel()) {
  console.log('✅ Sử dụng model đã train');
  if (!trainedModel.patternDict || Object.keys(trainedModel.patternDict).length === 0) {
    loadPatternCau().then(() => loadFormulaCau());
  }
} else {
  console.log('⚠️  Chưa có model. Bắt đầu train từ CSV...');
  loadAndTrainFromCSV().then(ok => {
    if (ok) {
      loadPatternCau().then(() => {
        loadFormulaCau().then(() => {
          fs.writeFileSync(TRAINED_MODEL_FILE, JSON.stringify(trainedModel));
        });
      });
    }
  });
}

app.listen(PORT, '0.0.0.0', () => {
  console.log(`\n🚀 Server: http://0.0.0.0:${PORT}`);
  console.log('🔥 Tài Xỉu VIP PRO ULTRA MAX 6.0 - 17K + 10K CẦU (BẢN FIX)');
  console.log('');
  console.log('📡 ENDPOINTS:');
  console.log('  GET /api/tx              - Dự đoán Hũ (có pattern_10)');
  console.log('  GET /api/tx/md5          - Dự đoán MD5 (có pattern_10)');
  console.log('  GET /api/history/dudoan  - 1000 phiên đúng/sai');
  console.log('  GET /api/history         - Lịch sử chi tiết');
  console.log('  GET /api/analysis        - Phân tích chi tiết');
  console.log('  GET /api/learning        - Stats AI');
  console.log('  GET /api/pattern10       - 10 cầu gần nhất');
  console.log('  GET /api/training-status - Trạng thái học');
  console.log('  GET /retrain             - Force retrain');
  console.log('  GET /reset-learning      - Reset learning');
  console.log('');
  console.log(`🧠 TRAINED: ${trainedModel.trained ? '✅' : '❌'}`);
  console.log(`📚 SAMPLES: ${trainedModel.totalSamples}`);
  console.log(`🎯 PATTERNS: ${Object.keys(trainedModel.patternDict || {}).length}`);
  console.log(`🔗 FORMULAS: ${Object.keys(trainedModel.formulaMap || {}).length}`);
  console.log('');
  console.log('💾 FILES: tiendat.json, tiendat1.json, dudoan_history.json,');
  console.log('           data_sessions.csv, cau.txt, cong_thuc_cau.txt, trained_model_17k.json');
  console.log(`👤 ID: ${AUTHOR_ID}\n`);
  
  startAutoSaveTask();
});
