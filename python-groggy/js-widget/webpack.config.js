const path = require('path');

module.exports = [
  {
    // Main npm package entry
    entry: './src/index.ts',
    output: {
      filename: 'index.js',
      path: path.resolve(__dirname, 'lib'),
      library: 'groggy-widgets',
      libraryTarget: 'amd',
      publicPath: '',
    },
    module: {
      rules: [
        {
          test: /\.ts$/,
          use: 'ts-loader',
          exclude: /node_modules/,
        },
        {
          test: /\.css$/,
          use: ['style-loader', 'css-loader'],
        },
      ],
    },
    resolve: {
      extensions: ['.ts', '.js'],
      alias: {
        '@groggy/core': path.resolve(__dirname, '..', 'python', 'groggy', 'widgets'),
      },
    },
    externals: [
      '@jupyter-widgets/base',
      '@jupyter-widgets/controls'
    ],
    devtool: 'source-map',
    mode: 'development'
  },

  {
    // Notebook extension
    entry: './src/extension.ts',
    output: {
      filename: 'index.js',
      path: path.resolve(__dirname, '..', 'python', 'groggy', 'nbextension'),
      libraryTarget: 'amd',
      publicPath: '',
    },
    module: {
      rules: [
        {
          test: /\.ts$/,
          use: 'ts-loader',
          exclude: /node_modules/,
        },
        {
          test: /\.css$/,
          use: ['style-loader', 'css-loader'],
        },
        {
          test: /\.js$/,
          use: 'source-map-loader',
          enforce: 'pre',
          exclude: /node_modules/,
        },
      ],
    },
    resolve: {
      extensions: ['.ts', '.js'],
      alias: {
        '@groggy/core': path.resolve(__dirname, '..', 'python', 'groggy', 'widgets'),
      },
    },
    externals: [
      '@jupyter-widgets/base',
      '@jupyter-widgets/controls'
    ],
    devtool: 'source-map',
    mode: 'development'
  },
  
  {
    // JupyterLab extension  
    entry: './src/plugin.ts',
    output: {
      filename: 'plugin.js',
      path: path.resolve(__dirname, 'lib'),
      libraryTarget: 'amd',
      publicPath: '',
    },
    module: {
      rules: [
        {
          test: /\.ts$/,
          use: 'ts-loader',
          exclude: /node_modules/,
        },
        {
          test: /\.css$/,
          use: ['style-loader', 'css-loader'],
        },
      ],
    },
    resolve: {
      extensions: ['.ts', '.js'],
      alias: {
        '@groggy/core': path.resolve(__dirname, '..', 'python', 'groggy', 'widgets'),
      },
    },
    externals: [
      '@jupyter-widgets/base',
      '@jupyter-widgets/controls',
      '@jupyterlab/application',
      '@jupyterlab/widgets'
    ],
    devtool: 'source-map',
    mode: 'development'
  }
];