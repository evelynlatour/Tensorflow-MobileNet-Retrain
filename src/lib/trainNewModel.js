import * as tf from '@tensorflow/tfjs'
import { getXs, getYs } from './createXYs'

export const compileModel = (numClasses) => {
  let model;
  model = tf.sequential({
    layers: [
      //This layer only performs a reshape so that we can use it in a dense layer
      tf.layers.flatten({ inputShape: [7, 7, 256] }),
      tf.layers.dense({
        units: 100,
        activation: 'relu',
        kernelInitializer: 'varianceScaling',
        useBias: true,
      }),
      tf.layers.dense({
        units: numClasses,
        kernelInitializer: 'varianceScaling', 
        useBias: false,
        activation: 'softmax', // multi-label: 'sigmoid'
      }),
    ],
  })

  model.compile({
    optimizer: tf.train.adam(0.0001),
    loss: 'categoricalCrossentropy', // for multi-label this becomes 'binary' since multilabel is viewed as a set of n, independent two-class problems
    metrics: ['accuracy'],
  })

  console.log('%c Multi-class Model Compiled:', 'color: #4295f4; font-weight: bold', model)  
  // model.summary()
  return model
}


export const fitModel = async (model, xs, ys, startingEpoch) => {
  await model.fit(xs, ys, {
    batchSize: 125, // dictates steps per epoch
    epochs: 250, // make sure this aligns w/ starting epoch in trainShuffledBatches
    initialEpoch: startingEpoch,
    shuffle: true,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log('%c Loss is: ', 'color: #ffb85b', logs.loss.toFixed(5))
      },
      onEpochBegin: async (epoch) => console.log('new epoch', epoch),
      onEpochEnd: async (epoch, logs) => {
        if ((epoch + 1) % 50 === 0) model.stopTraining = true
      }
    },
  })
  console.log('model', model) //model.model.history 
  return model 
}

// NOTE: tf.util provides more methods

export const trainShuffledBatches = async (trainingData, trainingLabels, numItems, labelKey, numClasses, batchSize) => {
  try {
    let batchStart = 0
    let currentBatch = 1
    let startingEpoch = 0
    let model = compileModel(numClasses)
    const totalBatches = numItems/batchSize
    const shuffleMap = tf.util.createShuffledIndices(numItems)
    console.log(shuffleMap)

    while (batchStart < numItems ) { 
      let dataBatch = []
      let labelBatch = []
      for (let i = batchStart; i < (batchStart + batchSize); i++){ 
        const newIndex = shuffleMap[i] // gives you the index to pull the image/label from
        dataBatch.push(trainingData[newIndex])
        labelBatch.push(trainingLabels[newIndex])
      }
      const xs = await getXs(dataBatch)
      const ys = await getYs(labelBatch, labelKey)
      const trainedModel = await fitModel(model, xs, ys, startingEpoch)
      console.log(`%c Batch ${currentBatch} of ${totalBatches} completed successfully`, 'color: #4295f4; font-weight: bold')
      xs.dispose();
      ys.dispose();
      batchStart += batchSize
      currentBatch++
      startingEpoch += 50 // make sure this aligns with epochs in fitModel()
      model = trainedModel // ensures that prev trained model gets passed through in next training round
    }

    return model
  } catch (err) {
    console.log(err)
  }
}