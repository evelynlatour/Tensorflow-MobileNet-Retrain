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

// tfjs docs recommend putting this inside a for loop and passing a "batch" of data at a time
// callbacks available: onTrainBegin, onTrainEnd, onEpochBegin, onEpochEnd, onBatchBegin, onBatchEnd
// will batch from your total number passed through - e.g. 
// batchSize 20 with 32 images means 2 steps per epoch
// batchSize 10 and 32 images means 4 steps per epoch (the last batch would only have 2 images tho)

export const fitModel = async (model, xs, ys, startingEpoch) => {
  console.log('~starting epoch~', startingEpoch)
  await model.fit(xs, ys, {
    batchSize: 7,
    epochs: 15, //10
    initialEpoch: startingEpoch,
    shuffle: true,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log('%c Loss is: ', 'color: #ffb85b', logs.loss.toFixed(5))
      },
      onEpochBegin: async (epoch) => console.log('new epoch', epoch),
      onEpochEnd: async (epoch, logs) => {
        if ((epoch + 1) % 3 === 0) model.stopTraining = true
      }
    },
  })
  // console.log('%c Fitting completed, your model will be printed below', 'color: #4295f4; font-weight: bold')
  console.log('model', model)
  // console.log(model) //model.model.history
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
      for (let i = batchStart; i < (batchStart + batchSize); i++){ // test training 2 batches of 65 each for 130 total
        const newIndex = shuffleMap[i] // gives you the index to pull the image/label from
        dataBatch.push(trainingData[newIndex])
        labelBatch.push(trainingLabels[newIndex])
      }
      const xs = await getXs(dataBatch)
      const ys = await getYs(labelBatch, labelKey)
      const trainedModel = await fitModel(model, xs, ys, startingEpoch)
      console.log(`%c Batch ${currentBatch} of ${totalBatches} completed successfully`, 'color: #4295f4; font-weight: bold')
      // console.log('this was the data batch', dataBatch)
      // console.log('this was the label batch', labelBatch)
      batchStart += batchSize
      currentBatch++
      startingEpoch += 10
      model = trainedModel
    }
    return model
  } catch (err) {
    console.log(err)
  }
}


export const predict = async (myModel, image, expectedLabel) => {
  try {
    //make a prediction through truncated mobilenet, getting the internal 
    //activation output from the model
    const activation = await predictFromTruncated(image) //[1,7,7,256]

    //make a prediction through our newly-trained model using this activation as input
    const prediction = await myModel.predict(activation)
    prediction.print()
    // Returns the index with the maximum probability. This number corresponds
    // to the class the model thinks is the most probable given the input.
    console.log(`prediction is index ${prediction.as1D().argMax().dataSync()[0]} from label object`)
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