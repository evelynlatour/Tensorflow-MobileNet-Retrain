import * as tf from '@tensorflow/tfjs'
import { predictFromTruncated } from './mobileNet'


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