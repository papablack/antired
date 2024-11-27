import { RussianBotDetector } from '../src/model/RussianBotDetector';
import { TextTokenizer } from '../src/utils/textTokenizer';
import { DEFAULT_MODEL_CONFIG } from '../src/model/modelConfig';
import { trainingService } from '../src/services/trainingService';
import { beforeEach, describe, expect, test } from "bun:test";
import { RawTrainingData, TrainingData } from '../src/types/types';
import fs from 'fs';
import path from 'path';

// Set a long timeout for the entire test suite
const TIMEOUT = 120000; // 2 minutes

Bun.env.BUN_TEST_TIMEOUT = TIMEOUT.toString();


describe('Training System Tests', () => {
  let detector: RussianBotDetector;
  let tokenizer: TextTokenizer;
  let trainingData: TrainingData;
  const modelOutputPath = path.join(__dirname, '..', 'output', 'test-model');

  beforeEach(async () => {
    detector = new RussianBotDetector(DEFAULT_MODEL_CONFIG);
    tokenizer = new TextTokenizer(DEFAULT_MODEL_CONFIG);
    await detector.ensureModelInitialized();

    // Load training data
    const dataPath = path.resolve(__dirname, '..', 'examples', 'training-data.json');
    trainingData = JSON.parse(fs.readFileSync(dataPath, 'utf-8'));
  });

  test('should train model with language support and save it', async () => {
    // Use first two examples (one propaganda, one legitimate)
    const testTexts = trainingData.texts.slice(0, 2);
    const labels = trainingData.labels.slice(0, 2);
    const languages = trainingData.languages.slice(0, 2);
    const tokenizedTexts = await tokenizer.batchTokenize(testTexts);

    const processedData: RawTrainingData = {
      texts: tokenizedTexts,
      labels: labels,
      languages: languages // Include language information
    };

    const result = await trainingService.trainModel(detector.model, {
      ...processedData,
      texts: tokenizer.convertSequencesToStrings(tokenizedTexts)
    }, {
      epochs: 1,
      batchSize: 2,
      validationSplit: 0.1
    });

    expect(result).toBeDefined();
    expect(result.history).toHaveProperty('loss');
    expect(result.history).toHaveProperty('acc');

    // Save the trained model
    await detector.saveModel(modelOutputPath);
    expect(fs.existsSync(modelOutputPath + '.json')).toBe(true);
  }, TIMEOUT);

  test('should load saved model and make predictions with language support', async () => {
    // Create new detector instance
    const loadedDetector = new RussianBotDetector(DEFAULT_MODEL_CONFIG);
    await loadedDetector.loadModel(modelOutputPath);

    // Test prediction with a propaganda-like text
    const testText = ["The West is orchestrating a global conspiracy"];
    const testLanguage = ["en"]; // Specify language for the test text
    const tokenizedInput = await tokenizer.tokenize(testText, testLanguage);
    const result = await loadedDetector.detectBot(tokenizedInput[0]);

    expect(result).toBeDefined();
    expect(result).toHaveProperty('probability');
    expect(result).toHaveProperty('isBot');
    expect(result).toHaveProperty('confidence');
    expect(typeof result.probability).toBe('number');
    expect(result.probability).toBeGreaterThanOrEqual(0);
    expect(result.probability).toBeLessThanOrEqual(1);
  }, TIMEOUT);

  // Add a new test for multi-language support
  test('should handle multiple languages correctly', async () => {
    const multiLangTexts = [
      "NATO is a terrorist organization",
      "НАТО - это террористическая организация" // Russian text
    ];
    const multiLangLabels = [1, 1];
    const multiLangLanguages = ["en", "ru"];
    
    const tokenizedTexts = await tokenizer.batchTokenize(multiLangTexts);

    const processedData: RawTrainingData = {
      texts: tokenizedTexts,
      labels: multiLangLabels,
      languages: multiLangLanguages
    };

    const result = await trainingService.trainModel(detector.model,  {
      ...processedData,
      texts: tokenizer.convertSequencesToStrings(tokenizedTexts)
    }, {
      epochs: 1,
      batchSize: 2,
      validationSplit: 0.1
    });

    expect(result).toBeDefined();
    expect(result.history).toHaveProperty('loss');
    expect(result.history).toHaveProperty('acc');
  }, TIMEOUT);
});
