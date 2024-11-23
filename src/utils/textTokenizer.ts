import * as tf from '@tensorflow/tfjs';
import * as use from '@tensorflow-models/universal-sentence-encoder';
import { Logger } from './logger';
import { BotDetectorConfig } from '../types/types';

export class TextTokenizer {
    private model: use.UniversalSentenceEncoder | null = null;
    private vocabulary: Map<string, number>;
    private config: BotDetectorConfig;

    constructor(config: BotDetectorConfig) {
        this.config = config;
        this.vocabulary = new Map();
        this.initializeModel();
    }

    private async initializeModel(): Promise<use.UniversalSentenceEncoder> {
        try {
            Logger.info('Loading Universal Sentence Encoder model...', 'blue');
            this.model = await use.load();
            Logger.info('Universal Sentence Encoder model loaded successfully', 'green');

            return this.model;
        } catch (error) {
            Logger.error('Failed to load Universal Sentence Encoder model:', error as Error);
            throw error;
        }
    }

    async tokenize(texts: string[]): Promise<number[][]> {
        if (!this.model) {
            this.model = await this.initializeModel(); // Auto-initialize if needed
        }

        try {         
            // Get embeddings using Universal Sentence Encoder
            const embeddings = await this.model.embed(texts);
            const embeddingArray = await embeddings.array();

            // Convert embeddings to fixed-length sequences
            const sequences = embeddingArray.map(embedding => 
                this.convertToFixedLengthSequence(
                    this.quantizeEmbedding(embedding)
                )
            );

            // Cleanup
            embeddings.dispose();

            return sequences;
        } catch (error) {
            Logger.error('Error during tokenization:', error as Error);
            throw error;
        }
    }

    private quantizeEmbedding(embedding: number[]): number[] {
        // Convert continuous embeddings to discrete tokens within vocabulary size
        return embedding.map(value => {
            // Scale and quantize the embedding values to fit within vocabulary size
            const scaled = Math.floor((value + 1) * (this.config.vocabSize / 2));
            return Math.max(0, Math.min(scaled, this.config.vocabSize - 1));
        });
    }

    private convertToFixedLengthSequence(sequence: number[]): number[] {
        if (sequence.length > this.config.maxSequenceLength) {
            // Truncate if longer than maxSequenceLength
            return sequence.slice(0, this.config.maxSequenceLength);
        } else if (sequence.length < this.config.maxSequenceLength) {
            // Pad with zeros if shorter than maxSequenceLength
            return [
                ...sequence,
                ...new Array(this.config.maxSequenceLength - sequence.length).fill(0)
            ];
        }
        return sequence;
    }

    async batchTokenize(texts: string[]): Promise<number[][]> {
        const batchSize = 32; // Adjust based on your memory constraints
        const results: number[][] = [];

        for (let i = 0; i < texts.length; i += batchSize) {
            const batch = texts.slice(i, i + batchSize);
            const batchTokens = await this.tokenize(batch);
            results.push(...batchTokens);
        }

        return results;
    }
}