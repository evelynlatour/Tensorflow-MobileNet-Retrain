'use strict';

// const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  entry: [`@babel/polyfill`,`./src/index.js`],
  output: {
    path: `${__dirname}/src`,
    publicPath: `/`,
    filename: `bundle.js`,
  },
  devServer: {
    contentBase: `./src`,
    watchContentBase: true,
  },
  module: {
    rules: [
      {
        test: /\.js$/,
        exclude: /(node_modules|bower_components)/,
        use: {
          loader: 'babel-loader',
          options: {
            presets: ['@babel/preset-env']
          }
        }
      },
      {
        test: /\.css$/,
        use: [`style-loader`, `css-loader`],
      },
      {
        test: /\.(gif|svg|jpg|png|jpeg)$/,
        use: `file-loader`,
      },
    ]
  },
  mode: `development`,
};