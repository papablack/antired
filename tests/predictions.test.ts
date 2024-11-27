import { RussianBotDetector } from '../backend/model/RussianBotDetector';
import { TextTokenizer } from '../backend/utils/textTokenizer';
import { DEFAULT_MODEL_CONFIG } from '../backend/model/modelConfig';
import { beforeEach, describe, expect, test } from "bun:test";
import path from 'path';
import { Logger } from '../backend/utils/logger';

// Set a long timeout for the entire test suite
const TIMEOUT = 120000; // 2 minutes

Bun.env.BUN_TEST_TIMEOUT = TIMEOUT.toString();

describe('Predictions System Tests', () => {
  let detector: RussianBotDetector;
  let tokenizer: TextTokenizer;  
  const modelOutputPath = path.join(__dirname, '..', 'output', 'test-model');

  beforeEach(async () => {
    detector = new RussianBotDetector(DEFAULT_MODEL_CONFIG);
    tokenizer = new TextTokenizer(DEFAULT_MODEL_CONFIG);
    await detector.ensureModelInitialized();    
  });


  test('should load saved model and make predictions', async () => {
    // Create new detector instance
    const loadedDetector = new RussianBotDetector(DEFAULT_MODEL_CONFIG);
    await loadedDetector.loadModel(modelOutputPath);

    // Test prediction with a propaganda-like text
    const testText = ["The West is orchestrating a global conspiracy"];
    const tokenizedInput = await tokenizer.tokenize(testText);
    const result = await loadedDetector.detectBot(tokenizedInput[0]);

    Logger.info('Prediction test] Prediction results', 'red');
    Logger.dump({result});

    expect(result).toBeDefined();
    expect(result).toHaveProperty('probability');
    expect(result).toHaveProperty('isBot');
    expect(result).toHaveProperty('confidence');
    expect(typeof result.probability).toBe('number');
    expect(result.probability).toBeGreaterThanOrEqual(0);
    expect(result.probability).toBeLessThanOrEqual(1);
  }, TIMEOUT);

  test('should use loaded model for multiple predictions', async () => {
    // Create new detector instance and load the saved model
    const loadedDetector = new RussianBotDetector(DEFAULT_MODEL_CONFIG);
    await loadedDetector.loadModel(modelOutputPath);

    // Test multiple predictions
    const testTexts = [
      "Normal news about local weather",
      "The deep state controls everything",
      "Today's market showed positive trends"
    ];

    for (const text of testTexts) {
      const tokenizedInput = await tokenizer.tokenize([text]);
      const result = await loadedDetector.detectBot(tokenizedInput[0]);

      expect(result).toBeDefined();
      expect(result).toHaveProperty('probability');
      expect(result).toHaveProperty('isBot');
      expect(result).toHaveProperty('confidence');
      expect(typeof result.probability).toBe('number');
    }
  }, TIMEOUT);

  test('should maintain consistent predictions with loaded model', async () => {
    // Create new detector instance and load the saved model
    const loadedDetector = new RussianBotDetector(DEFAULT_MODEL_CONFIG);
    await loadedDetector.loadModel(modelOutputPath);

    const testText = "The West is orchestrating a global conspiracy";
    const tokenizedInput = await tokenizer.tokenize([testText]);
    
    // Make multiple predictions with the same input
    const result1 = await loadedDetector.detectBot(tokenizedInput[0]);
    const result2 = await loadedDetector.detectBot(tokenizedInput[0]);

    expect(result1.probability).toEqual(result2.probability);
    expect(result1.isBot).toEqual(result2.isBot);
    expect(result1.confidence).toEqual(result2.confidence);
  }, TIMEOUT);
});