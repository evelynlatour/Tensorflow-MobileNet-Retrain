import * as tf from '@tensorflow/tfjs'
import { getXs, getYs, predictFromTruncated } from './mobileNet'


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

// callbacks available: onTrainBegin, onTrainEnd, onEpochBegin, onEpochEnd, onBatchBegin, onBatchEnd
// batchSize 20 with 32 images means 2 steps per epoch
// batchSize 10 and 32 images means 4 steps per epoch (the last batch would only have 2 images)

export const fitModel = async (model, xs, ys, startingEpoch) => {
  await model.fit(xs, ys, {
    batchSize: 10, // dictates steps per epoch
    epochs: 100, // make sure this aligns w/ starting epoch in trainShuffledBatches
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

/* tf.util provides more methods! */


export const trainShuffledBatches = async (trainingData, trainingLabels, numItems, labelKey, numClasses, batchSize) => {
  try {
    let batchStart = 0
    let currentBatch = 1
    let startingEpoch = 0
    let model = compileModel(numClasses)
    const totalBatches = numItems/batchSize
    const shuffleMap = tf.util.createShuffledIndices(numItems)
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


export const listModelsInLocalStorage = async () => {
  console.log('%c Models available in the browser...', 'color: #63DFFF')
  console.log(await tf.io.listModels())
}

export const removeModelFromLocalStorage = async (modelName) => {
  console.log(`%c Removing model "${modelName}" from local storage`, 'color: #f4425c')
  tf.io.removeModel(`indexeddb://${modelName}`)
}


export const loadCustomModel = async (modelName) => {
  console.log(`%c Loading ${modelName}`, 'color: #49FFE0')
  const customModel = await tf.loadModel(`indexeddb://${modelName}`)
  return customModel
} 


export const predict = async (modelName, image, labelKey, expectedLabel) => {
  try {
    const myModel = await loadCustomModel(modelName)

    //make a prediction through truncated mobilenet, getting the internal 
    //activation output from the model
    const activation = await predictFromTruncated(image) //[1,7,7,256]

    //make a prediction through our newly-trained model using this activation as input
    const prediction = await myModel.predict(activation)
    prediction.print()
    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    console.log(labelKey)
    const predictedLabelIndex = prediction.as1D().argMax().dataSync()[0]
    console.log(`${modelName} model predicts ${Object.keys(labelKey)[predictedLabelIndex]} expected ${expectedLabel}`)
    return prediction.as1D().argMax();
  } catch (err) {
    console.log(err)
  }
}




/*
Multi-class problems: classes are mutually exclusive
Multi-label problems: classes not mutually exclusive (they are now called labels too). 
this classification problem assigns a set of target labels to each sample (e.g. image)
each label represents a different classification task, but the tasks
are in some way related so there is benefit in tackling them together
(I think you may want a multioutput regression with this?)

tf.nn.sigmoid_cross_entropy_with_logits

Sigmoid, unlike softmax don't give probability distribution around nclasses as output, 
but independent probabilities.

Measures the probability error in discrete classification tasks in which each class is independent 
and not mutually exclusive. For instance, one could perform multilabel classification where a 
picture can contain both an elephant and a dog at the same time.

logits and labels must have the same type and shape.

softmax_cross_entropy_with_logits: classes are mutually exclusive, but probabilities need not be
*/