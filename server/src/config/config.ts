import { IAppConfig } from "@rws-framework/server";
import { ConfigHelper } from "./helper";

import httpRoutes from '../routing/routes';
import { MLModule } from "../app/ml.module";


export interface IMLOpts extends IAppConfig {
    anthropic_model: string
    anthropic_api_key: string
    cohere_api_key: string
}

export default (): IMLOpts => { 
    const configHelper = new ConfigHelper();

    return {        
        anthropic_model: 'claude-3-5-sonnet-20240620',
        anthropic_api_key: configHelper.get('ANTHROPIC_KEY'),
        cohere_api_key: configHelper.get('COHERE_KEY'),        
        features: {
            routing_enabled: true,
            ws_enabled: true,
            ssl: false,
            auth: false
        },
        user_class: null,
        user_models: [],
        port: 3001,        
        ws_port: 3002,        
        domain: 'localhost',
        cors_domain: '*',
        secret_key: configHelper.get('APP_SECRET'),               
        modules: [
            MLModule
        ],
        ws_routes: {              
        },
        http_routes: httpRoutes,
        mongo_db: '',
        mongo_url: '',
        ssl_cert: '',
        ssl_key: ''
    }
}