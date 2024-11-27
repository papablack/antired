import { serverInit, ConsoleService, IAuthUser, IDbUser } from "@rws-framework/server";
import config from './config/config';
import JWTUser from "./user/model";
import { MLModule } from "./app/ml.module";

const onAuthorize = async <T extends IDbUser>(user: T, authorizationScope: 'ws' | 'http'): Promise<void> => {
    const jwtUser: JWTUser = user as any;
    console.log('RWS AUTHORIZED', authorizationScope, jwtUser);
};

async function main(): Promise<any> {                
    return await serverInit(MLModule, config, { 
        authorization: true, 
        transport: 'websocket', 
        onAuthorize 
    });    
}

main().then((servers: any) => {
    console.log('Server started successfully');
}).catch((e) => {
    console.error(e);
});