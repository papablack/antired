import winston from 'winston';
import chalk, { ChalkInstance } from 'chalk';
import path from 'path';

// Define custom log levels
const customLevels = {
    error: 0,
    info: 1,
    training: 2,
    debug: 3
};

// Define colors for each level
const customColors = {
    error: 'red',
    info: 'white',
    training: 'cyan',
    debug: 'gray'
};

const colorParts: {[key: string]: ChalkInstance} = {
    logDate: chalk.rgb(255, 140, 0)
}

// Add colors to winston
winston.addColors(customColors);

const winstonLogger = winston.createLogger({
    levels: customLevels,
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.printf(({ level, message, timestamp }) => {
            return `${timestamp} [${level}]: ${message}`;
        })
    ),
    transports: [
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.timestamp(),
                winston.format.printf(({ level, message, timestamp }) => {
                    // Color the timestamp in magenta and level based on its type
                    const coloredTimestamp = colorParts.logDate(timestamp);
                    const levelColor = customColors[level];
                    const coloredLevel = chalk[levelColor](`[${level}]`);
                    return `${coloredTimestamp} ${coloredLevel}: ${message}`;
                })
            )
        }),
        new winston.transports.File({ 
            filename: path.resolve(process.cwd(), 'logs/debug.log')
        })
    ]
});

export type LoggerColors = 'red' | 'blue' | 'yellow' | 'green' | 'cyan' | 'magenta';

export class Logger {
    private constructor() {}

    static colors: Record<LoggerColors, ChalkInstance> = {
        red: chalk.red,
        blue: chalk.blue,
        yellow: chalk.yellow,
        green: chalk.green,
        cyan: chalk.cyan,
        magenta: chalk.magenta,        
    };

    static dump(object: any): void {
        console.log(object)
    }

    static info(message: string, color?: LoggerColors): void {
        const coloredMessage = color ? 
            Logger.colors[color](message) : 
            message;
        winstonLogger.info(coloredMessage);
    }

    static error(message: string, error?: Error): void {
        const errorMessage = error ? 
            `${message}: ${error.message}\n${error.stack}` : 
            message;
        
        winstonLogger.error(Logger.colors.red(errorMessage));
    }

    static training(epoch: number, logs: { loss: number; acc: number }): void {
        const message = 
            `Epoch ${epoch + 1} - ` +
            `loss: ${logs?.loss.toFixed(4)} - ` +
            `accuracy: ${logs?.acc.toFixed(4)}`;

        winstonLogger.log('training', Logger.colors.cyan(message));
    }
}