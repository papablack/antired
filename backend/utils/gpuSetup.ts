import * as tf from '@tensorflow/tfjs-node';
import * as tfnode from '@tensorflow/tfjs-node-gpu';
import { Logger } from './logger';

export async function setupGPU() {
    try {
        // Initialize tensorflow with GPU support
        await tf.setBackend('webgl'); // or 'tensorflow'
        tfnode.enableProdMode();
        
        // Check if GPU is available
        const gpuAvailable = tf.getBackend() === 'webgl';
        
        if (gpuAvailable) {
            Logger.info('GPU is available and will be used for computations', 'green');
            // Get device information
            const memory = (tf as any).memory();
            Logger.info('GPU Memory Info:', 'blue');
            Logger.dump({
                numBytes: memory.numBytes,
                numTensors: memory.numTensors,
                numDataBuffers: memory.numDataBuffers,
                unreliable: memory.unreliable
            });
        } else {
            Logger.info('No GPU detected, falling back to CPU', 'yellow');
        }

        // Log current backend
        Logger.info(`Current backend: ${tf.getBackend()}`, 'blue');
    } catch (error) {
        Logger.error('Error setting up GPU:', error as Error);
        throw error;
    }
}