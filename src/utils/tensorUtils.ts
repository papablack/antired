import * as tf from '@tensorflow/tfjs';

export class TensorUtils {
    static prepareInputData(texts: number[][]): tf.Tensor2D {
        return tf.tensor2d(texts);
    }

    static prepareLabels(labels: number[]): tf.Tensor2D {
        return tf.tensor2d(labels, [labels.length, 1]);
    }

    static async tensorToNumber(tensor: tf.Tensor): Promise<number> {
        const data = await tensor.data();
        return data[0];
    }

    static dispose(...tensors: tf.Tensor[]): void {
        tensors.forEach(tensor => tensor.dispose());
    }
}