import * as tf from '@tensorflow/tfjs'
import { predictFromTruncated } from './mobileNet'

const numClasses = 2; // already handles multi-class
let model;

export const trainModel = async (xs, ys) => {
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

  await model.compile({
    optimizer: tf.train.adam(0.0001),
    loss: 'categoricalCrossentropy', // this becomes binary since multilabel is viewed as a set of n, independent two-class problems
    metrics: ['accuracy'],
  })

  // may want to return/save your model here and then do fit() func in a sep func because you'll
  // need to call it once every batch (the whole dataset prob can't fit into memory)

  await model.fit(xs, ys, {
    //batch size of 1 assumed if not specified?
    epochs: 10,
    shuffle: true,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        console.log('Loss is: ', logs.loss.toFixed(5))
      },
    },
  })
  // model.print()
  return model
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