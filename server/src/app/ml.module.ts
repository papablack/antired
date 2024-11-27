import { Module  } from '@nestjs/common';
import {  UtilsService, RWSFillService, RWSModule, AuthService, ConsoleService, AppConfigService } from '@rws-framework/server';
// import { AnthropicService } from '../services/AnthropicService';
import { IMLOpts } from '../config/config';
import { MLController } from '../controllers/ml.controller';


@Module({})
export class MLModule {
  static forRoot(cfg: IMLOpts){    
    return {
      module: MLModule,
      imports: [
        RWSModule.forRoot(cfg)
      ],
      controllers: [
        MLController
      ],
      providers: [
        UtilsService,
        RWSFillService,
      ],
    }
  }
}