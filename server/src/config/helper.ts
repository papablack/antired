import { readIniFile } from 'read-ini-file';
import path from 'path';
import fs from 'fs';
import {Injectable} from "@rws-framework/server/nest";
import { rwsPath } from "@rws-framework/console"
import dotenv from 'dotenv';

type TheConfig = { [key: string]: string };

Injectable()
class ConfigHelper {
  private envData: TheConfig = {}; // Add type assertion here
  private envFilePath: string | null = null;

  constructor() {
    this.initEnv();
  }

  private initEnv(){
    const cfgDir = this.findConfigFileFolder();
    
    this.envFilePath = `${cfgDir}/.env`;    

    if(!fs.existsSync(cfgDir + '/.env')){
      throw new Error(`Config file ${cfgDir}/.env not found`);
    }

    this.envData = this.parseDotEnv(this.envFilePath);

    if(fs.existsSync(cfgDir + '/.env.local')){
      this.envFilePath = cfgDir + '/.env.local';
      this.envData = Object.assign(this.envData, this.parseDotEnv(this.envFilePath));      
    }
  }

  private parseDotEnv(path: string): TheConfig {
    try {
      const result = dotenv.config({ path: path });
      if (result.error) {
        throw result.error;
      }
      return result.parsed as TheConfig;
    } catch (error: Error | any) {
      throw new Error(`Error parsing .env file: ${error.message}`);
    }    
  }

  public findConfigFileFolder(folder_path:string | null = null): string | null
  {
    const rwsDir = rwsPath.findPackageDir(__dirname);

    return path.resolve(rwsDir, '..');
  }

  private mergeDeep<T>(target: T | any, source: T  | any): T 
  {
    const isObject = (obj: any) => obj && typeof obj === 'object';

    if (!isObject(target) || !isObject(source)) {
        return source;
    }

    Object.keys(source).forEach(key => {
        const targetValue = target[key];
        const sourceValue = source[key];

        if (Array.isArray(targetValue) && Array.isArray(sourceValue)) {
            target[key] = targetValue.concat(sourceValue);
        } else if (isObject(targetValue) && isObject(sourceValue)) {
            target[key] = this.mergeDeep(Object.assign({}, targetValue), sourceValue);
        } else {
            target[key] = sourceValue;
        }
    });

    return target;
  }
  

  get = (key: keyof TheConfig) => {   
    return this.envData[key];
  };
}

export { ConfigHelper };