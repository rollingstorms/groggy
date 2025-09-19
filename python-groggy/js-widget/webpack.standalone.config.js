const path = require('path');

module.exports = {
  // Standalone bundle for embedding in HTML files
  entry: './src/core/index.js',
  output: {
    filename: 'groggy-viz-core.standalone.js',
    path: path.resolve(__dirname, 'lib'),
    library: 'GroggyVizCore',
    libraryTarget: 'umd',  // Universal Module Definition - works in browsers
    globalObject: 'this',
    umdNamedDefine: true
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /node_modules/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env']
          }
        }
      },
      {
        test: /\.css$/,
        use: ['style-loader', 'css-loader'],
      },
    ],
  },
  resolve: {
    extensions: ['.js'],
  },
  devtool: 'source-map',
  mode: 'production',  // Minified for embedding
  optimization: {
    minimize: true
  }
};