import * as tf from '@tensorflow/tfjs';
import { TrainingData, TrainingOptions } from '../types/types';
import { Logger } from '../utils/logger';
import { TRAINING_DEFAULTS } from '../model/modelConfig';
import { TextTokenizer } from '../utils/textTokenizer';
import { DEFAULT_MODEL_CONFIG } from '../model/modelConfig';

export type TFLogType = { loss: number; acc: number };

export class TrainingHelper {
    private static tokenizer: TextTokenizer;

    private constructor() {
        TrainingHelper.tokenizer = new TextTokenizer(DEFAULT_MODEL_CONFIG);
    }

    static async trainModel(
        model: tf.LayersModel,
        data: TrainingData,
        options: TrainingOptions = {}
    ): Promise<tf.History> {
        try {
            const { texts, labels, languages } = data;
            const {
                epochs = TRAINING_DEFAULTS.epochs,
                batchSize = TRAINING_DEFAULTS.batchSize,
                validationSplit = TRAINING_DEFAULTS.validationSplit
            } = options;
    
            // Add input validation and debugging
            Logger.dump({
                textsLength: texts.length,
                firstText: texts[0],
                firstTextType: typeof texts[0],
                isArray: Array.isArray(texts[0])
            });
    
            if (!Array.isArray(texts) || texts.length === 0) {
                throw new Error('Invalid input texts: must be a non-empty array');
            }
    
            // Convert texts to arrays of numbers if they're strings
            const processedTexts = texts.map((text, idx) => {
                if (typeof text === 'string') {
                    // Split string and convert to numbers
                    return text.split(' ').map(num => parseInt(num, 10));
                }
                return text;
            });
    
            // Ensure texts are properly padded/truncated to match maxSequenceLength
            const paddedTexts = processedTexts.map((text, idx) => {
                // Ensure text is an array of numbers
                if (!Array.isArray(text)) {
                    Logger.error(`Invalid text at index ${idx}: ` + text);
                    throw new Error(`Text at index ${idx} must be an array of numbers`);
                }
    
                // Convert any string numbers to actual numbers
                const numericText = text.map(val => {
                    const num = Number(val);
                    if (isNaN(num)) {
                        Logger.error(`Invalid token value at index ${idx}: ` + val);
                        throw new Error(`Invalid token value at index ${idx}: ${val}`);
                    }
                    return num;
                });
    
                // Create array of maxSequenceLength filled with padding token (0)
                const paddedArray = new Array(DEFAULT_MODEL_CONFIG.maxSequenceLength).fill(0);
                
                // Copy tokens from input text, truncating if longer than maxSequenceLength
                for (let i = 0; i < Math.min(numericText.length, DEFAULT_MODEL_CONFIG.maxSequenceLength); i++) {
                    paddedArray[i] = numericText[i];
                }
                
                return paddedArray;
            });
    
            // Debug padded texts
            Logger.dump({
                paddedLength: paddedTexts.length,
                firstPaddedText: paddedTexts[0],
                paddedShape: [paddedTexts.length, paddedTexts[0].length]
            });
    
            // Create input tensor
            const xs = tf.tidy(() => {
                if (languages) {
                    // Adjust text tensor to account for language feature
                    const adjustedMaxLength = DEFAULT_MODEL_CONFIG.maxSequenceLength - 1; // Reduce by 1 to make room for language
                    const paddedTexts = processedTexts.map(text => text.slice(0, adjustedMaxLength));
                    
                    const textTensor = tf.tensor2d(paddedTexts, [paddedTexts.length, adjustedMaxLength]);
                    const languageIds = languages.map(lang => {
                        const id = this.tokenizer.getLanguageId(lang);
                        Logger.info(`Language ${lang} mapped to ID: ${id}`);
                        return id;
                    });
                    const languageTensor = tf.tensor2d(languageIds, [languageIds.length, 1]);
                    
                    // Debug tensors
                    Logger.dump({
                        textTensor: textTensor.shape,
                        languageTensor: languageTensor.shape
                    });
                    
                    // Concatenate along the second dimension to maintain the expected shape
                    return tf.concat([textTensor, languageTensor], 1);
                }
                return tf.tensor2d(paddedTexts);
            });
            
    
            const ys = tf.tidy(() => {
                return tf.tensor2d(labels, [labels.length, 1]);
            });

            Logger.info('Starting model training...');
            Logger.info(`Training with ${languages ? 'language support' : 'no language support'}`);

            const history = await model.fit(xs, ys, {
                epochs,
                batchSize,
                validationSplit,
                callbacks: {
                    onEpochEnd: (epoch, logs) => {                
                        return Logger.training(epoch, logs as TFLogType)
                    }
                }
            });
            
            Logger.info('Training completed successfully!', 'green');
            
            // Cleanup tensors
            tf.dispose([xs, ys]);
            return history;
        } catch (error) {
            Logger.error('Error during training:', error as Error);
            throw error;
        }
    }
}