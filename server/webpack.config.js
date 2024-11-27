const path = require('path');

const  RWSWebpackWrapper  = require('@rws-framework/server/rws.webpack.config');

const executionDir = process.cwd();

module.exports = RWSWebpackWrapper({
  dev: false,  
  tsConfigPath: executionDir + '/tsconfig.json',
  entry: `${executionDir}/src/index.ts`,
  executionDir: executionDir,  
  outputDir:  path.resolve(executionDir, 'build'),
  outputFileName: 'antired.rws.js'
});