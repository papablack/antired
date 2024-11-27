import * as tf from '@tensorflow/tfjs';
import { BotDetectorConfig, PredictionResult, TrainingData } from '../types/types';
import { DEFAULT_MODEL_CONFIG, MODEL_LAYERS_CONFIG } from './modelConfig';
import { trainingService } from '../services/trainingService';
import { TensorUtils } from '../utils/tensorUtils';
import { Logger } from '../utils/logger';
import fs from 'fs';
import path from 'path';

export class RussianBotDetector {
    public model: tf.LayersModel;
    private config: BotDetectorConfig;
    private isInitialized: boolean = false;

    constructor(config: BotDetectorConfig = DEFAULT_MODEL_CONFIG) {
        this.config = config;
        this.initializeModel();
    }

    private async initializeModel(): Promise<void> {
        try {
            this.model = this.buildModel();
            this.isInitialized = true;
            Logger.info('Model initialized successfully');
        } catch (error) {
            Logger.error('Failed to initialize model:', error as Error);
            throw new Error('Model initialization failed');
        }
    }

    private buildModel(): tf.LayersModel {
        const model = tf.sequential();

        // Input layer with embedding
        model.add(tf.layers.embedding({
            inputDim: this.config.vocabSize,
            outputDim: this.config.embeddingDim,
            inputLength: this.config.maxSequenceLength,
            name: 'embedding_layer'
        }));

        // Conv1D layer
        model.add(tf.layers.conv1d({
            filters: MODEL_LAYERS_CONFIG.conv.filters,
            kernelSize: MODEL_LAYERS_CONFIG.conv.kernelSize,
            padding: 'same',
            activation: 'relu',
            name: 'conv1d_layer'
        }));

        // MaxPooling1D layer
        model.add(tf.layers.maxPooling1d({
            poolSize: 2,
            name: 'maxpool_layer'
        }));

        // First LSTM layer
        model.add(tf.layers.lstm({
            units: MODEL_LAYERS_CONFIG.lstm.firstLayer,
            returnSequences: true,
            name: 'lstm_layer_1'
        }));

        // Second LSTM layer
        model.add(tf.layers.lstm({
            units: MODEL_LAYERS_CONFIG.lstm.secondLayer,
            name: 'lstm_layer_2'
        }));

        // Dense layer with dropout
        model.add(tf.layers.dense({
            units: MODEL_LAYERS_CONFIG.dense.units,
            activation: 'relu',
            name: 'dense_layer'
        }));

        model.add(tf.layers.dropout({
            rate: MODEL_LAYERS_CONFIG.dense.dropout,
            name: 'dropout_layer'
        }));

        // Output layer
        model.add(tf.layers.dense({
            units: 1,
            activation: 'sigmoid',
            name: 'output_layer'
        }));

        // Compile the model
        model.compile({
            optimizer: tf.train.adam(0.001),
            loss: 'binaryCrossentropy',
            metrics: ['accuracy']
        });

        return model;
    }

    async ensureModelInitialized(): Promise<void> {
        if (!this.isInitialized || !this.model) {
            await this.initializeModel();
        }
    }

    async detectBot(text: number[]): Promise<PredictionResult> {
        await this.ensureModelInitialized();
        
        if (!this.model) {
            throw new Error('Model is not initialized');
        }

        // Ensure text is properly shaped for the model
        const input = tf.tidy(() => {
            // Reshape input to 3D: [batch_size, timesteps, features]
            return tf.tensor2d([text], [1, this.config.maxSequenceLength]);
        });

        try {
            const prediction = this.model.predict(input) as tf.Tensor;
            const probability = await TensorUtils.tensorToNumber(prediction);
            
            // Cleanup tensors
            tf.dispose([prediction, input]);

            return {
                probability,
                isBot: probability > 0.5,
                confidence: this.getConfidenceLevel(probability)
            };
        } catch (error) {
            Logger.error('Error during bot detection:', error as Error);
            throw error;
        }
    }

    private getConfidenceLevel(probability: number): 'high' | 'medium' | 'low' {
        if (probability > 0.8 || probability < 0.2) return 'high';
        if (probability > 0.65 || probability < 0.35) return 'medium';
        return 'low';
    }

    async train(data: TrainingData, options = {}): Promise<tf.History> {
        await this.ensureModelInitialized();
        
        if (!this.model) {
            throw new Error('Model is not initialized');
        }

        return trainingService.trainModel(this.model, data, options);
    }

    async saveModel(modelPath: string): Promise<void> {
        try {
            const dir = path.dirname(modelPath);
            if (!fs.existsSync(dir)) {
                fs.mkdirSync(dir, { recursive: true });
            }                        
    
            // Save the model weights as a binary file
            const artifacts = await this.model.save({
                save: async (artifacts: tf.io.ModelArtifacts) => {
                    // Save model topology
                    fs.writeFileSync(`${modelPath}.json`, JSON.stringify(artifacts.modelTopology));
                    
                    // Save weights
                    const weightData = new Uint8Array(artifacts.weightData as ArrayBuffer);
                    fs.writeFileSync(`${modelPath}.weights.bin`, weightData);
                    
                    // Save weight specs
                    fs.writeFileSync(`${modelPath}.weights.json`, JSON.stringify(artifacts.weightSpecs));
                    
                    return {
                        modelArtifactsInfo: {
                            dateSaved: new Date(),
                            modelTopologyType: 'JSON'
                        }
                    };
                }
            });
            
            Logger.info(`Model saved to ${modelPath}`, 'green');
        } catch (error) {
            Logger.error('Error saving model:', error as Error);
            throw error;
        }
    }
    
    async loadModel(modelPath: string): Promise<void> {
        try {
            // Custom loading function
            this.model = await tf.loadLayersModel({
                load: async () => {
                    // Load model topology
                    const modelTopology = JSON.parse(fs.readFileSync(`${modelPath}.json`, 'utf8'));
                    // Load weights file as a Buffer and convert to ArrayBuffer
                    const weightData = fs.readFileSync(`${modelPath}.weights.bin`).buffer;
                    // Load weight specs
                    const weightSpecs = JSON.parse(fs.readFileSync(`${modelPath}.weights.json`, 'utf8'));
                    
                    return {
                        modelTopology,
                        weightSpecs,
                        weightData: weightData as ArrayBuffer, // Convert Buffer to ArrayBuffer
                    };
                }
            });
        
            
            this.isInitialized = true;
            Logger.info(`Model loaded from ${modelPath}`);
        } catch (error) {
            Logger.error('Error loading model:', error as Error);
            throw error;
        }
    }
}