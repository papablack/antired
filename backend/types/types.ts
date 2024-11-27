export interface BotDetectorConfig {
    vocabSize: number;
    maxSequenceLength: number;
    embeddingDim: number;
}

export interface TrainingData {
    texts: string[];
    labels: number[];
    languages: string[];  // Add language support
}

export interface RawTrainingData {
    texts: number[][];
    labels: number[];
    languages: string[];  // Add language support
}

export interface TrainingOptions {
    epochs: number;
    batchSize: number;
    validationSplit: number;
    verbose?: boolean
    shuffle?: boolean
    callbacks?: (()=>void)[]
}

export interface PredictionResult {
    probability: number;
    isBot: boolean;
    confidence: 'high' | 'medium' | 'low';
}

export interface ModelParams {
    train?: boolean;
    run?: boolean;
    data?: TrainingData;  // Update to use RawTrainingData
}