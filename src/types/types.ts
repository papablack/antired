export interface BotDetectorConfig {
    vocabSize: number;
    maxSequenceLength: number;
    embeddingDim: number;
}

export interface TrainingData {
    texts: number[][];  // Allow both number arrays and strings
    labels: number[];
}

export interface TrainingOptions {
    epochs?: number;
    batchSize?: number;
    validationSplit?: number;
}

export interface PredictionResult {
    probability: number;
    isBot: boolean;
    confidence: 'high' | 'medium' | 'low';
}

export interface ModelParams {
    train?: boolean;
    run?: boolean;
    data?: {
        texts: string[];
        labels: number[];
    };
}
