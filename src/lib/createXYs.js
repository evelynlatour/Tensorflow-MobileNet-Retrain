import * as tf from '@tensorflow/tfjs'
import { loadTruncatedMobileNet, formatImage }  from './mobileNet'

/* Function to get the xs --> all the activations from the truncated MobileNet 
model after passing through our training data  */
export const getXs = async (images) => {
  try {
    let xs;
    const truncatedModel = await loadTruncatedMobileNet()
    console.log('%c Converting images to tensors...', 'color: #b159ff; font-weight: bold')
    for (let i = 0; i < images.length; i++) {
      const image = images[i];
      const imageToPredict = await formatImage(image)
      const processedImage = await truncatedModel.predict(imageToPredict)

      if (!xs) { // handle first run through
        xs = tf.keep(processedImage)
      } else { //handle all others
        const prevXs = xs;
        xs = tf.keep(prevXs.concat(processedImage, 0));
        prevXs.dispose();
      }
    }
    // xs.print();
    console.log('%c These are your xs: ', 'color: #ffb85b; font-weight: bold' , xs) // shape of Xs: [10,7,7,256] where 10 is the batch size (from testing 5 red & 5 blue)
    return xs;
  } catch (err) {
    console.log(err)
  }
}


/* Get the ys --> the labels for all of the collected data as a "one hot" representation */
// labelIndex is the index at which there is a 1
// oneHot(3, 5) => [0,0,0,1,0]

export const oneHot = (labelIndex, numClasses) => { 
  return tf.tidy(() => tf.oneHot(tf.tensor1d([labelIndex]).toInt(), numClasses));
};

// Use only if you do not have a predefined label object (e.g. for an array like red/blue testing)
export const labelObjectMaker = (labels) => {
  let labelObj = {};
  labels.forEach(label => {
    if (labelObj[label] === undefined) {
      labelObj[label] = Object.keys(labelObj).length
    }
  })
  console.log('this is the label object key: ', labelObj)
  return labelObj
}

export const getYs = (labels, labelKey) => {
  // const classes = labelObjectMaker(labels); // see note above on WHEN to use this func
  const classes = labelKey
  const classLength = Object.keys(classes).length;
  let ys;

  labels.forEach(label => {
    const labelIndex = classes[label];
    const y = oneHot(labelIndex, classLength);
    if (!ys) {
      ys = tf.keep(y)
    } else {
      const prevYs = ys;
      ys = tf.keep(prevYs.concat(y, 0))
      prevYs.dispose();
      y.dispose();
    }
  })
  // ys.print();
  console.log('%c These are your ys: ', 'color: #ffb85b; font-weight: bold' , ys)
  return ys;
}