import * as tf from '@tensorflow/tfjs';
import { BotDetectorConfig, PredictionResult, TrainingData } from '../types/types';
import { DEFAULT_MODEL_CONFIG, MODEL_LAYERS_CONFIG } from './modelConfig';
import { TrainingService } from '../services/trainingService';
import { TensorUtils } from '../utils/tensorUtils';
import { Logger } from '../utils/logger';

export class RussianBotDetector {
    private model: tf.LayersModel;
    private config: BotDetectorConfig;

    constructor(config: BotDetectorConfig = DEFAULT_MODEL_CONFIG) {
        this.config = config;
        this.model = this.buildModel();
    }

    private buildModel(): tf.LayersModel {
        const model = tf.sequential();

        // Embedding
        model.add(tf.layers.embedding({
            inputDim: this.config.vocabSize,
            outputDim: this.config.embeddingDim,
            inputLength: this.config.maxSequenceLength,
            name: 'warstwa_wektorow_slow'
        }));

        // Conv1D
        model.add(tf.layers.conv1d({
            filters: MODEL_LAYERS_CONFIG.conv.filters,
            kernelSize: MODEL_LAYERS_CONFIG.conv.kernelSize,
            padding: 'same',
            activation: 'relu',
            name: 'wykrywacz_propagandy_lokalnej'
        }));

        // MaxPooling1D
        model.add(tf.layers.maxPooling1d({
            poolSize: 2,
            name: 'reduktor_szumu_propagandowego'
        }));

        // LSTM layers
        model.add(tf.layers.lstm({
            units: MODEL_LAYERS_CONFIG.lstm.firstLayer,
            returnSequences: true,
            name: 'analizator_sekwencji'
        }));

        model.add(tf.layers.lstm({
            units: MODEL_LAYERS_CONFIG.lstm.secondLayer,
            name: 'analizator_glebokiej_propagandy'
        }));

        // Dense + Dropout
        model.add(tf.layers.dense({
            units: MODEL_LAYERS_CONFIG.dense.units,
            activation: 'relu',
            name: 'klasyfikator_wstepny'
        }));

        model.add(tf.layers.dropout({
            rate: MODEL_LAYERS_CONFIG.dense.dropout,
            name: 'eliminator_przeuczenia'
        }));

        // Output layer
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            name: 'final_decyzja_czy_ruski_bot'
        }));

        model.compile({
            optimizer: tf.train.adam(),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    async detectBot(text: number[]): Promise<PredictionResult> {
        const input = TensorUtils.prepareInputData([text]);
        try {
            const prediction = await this.model.predict(input) as tf.Tensor;
            const probability = await TensorUtils.tensorToNumber(prediction);
            return {
                probability,
                isBot: probability > 0.5,
                confidence: this.getConfidenceLevel(probability)
            };
        } catch (error) {
            Logger.error('Błąd podczas wykrywania bota:', error as Error);
            throw error;
        } finally {
            TensorUtils.dispose(input);
        }
    }

    private getConfidenceLevel(probability: number): 'high' | 'medium' | 'low' {
        if (probability > 0.8 || probability < 0.2) return 'high';
        if (probability > 0.65 || probability < 0.35) return 'medium';
        return 'low';
    }

    async train(...args: Parameters<typeof TrainingService.trainModel>) {
        return TrainingService.trainModel(...args);
    }
}