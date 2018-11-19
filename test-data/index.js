const importAllAsObject = req => {
  let images = {}
  req.keys().map(item => {
    images[item.replace(/.\//, '').replace(/\.(png|jpe?g)$/, '')] = req(item)
  })
  return images
}

const testData = importAllAsObject(require.context('./images', false, /\.(png|jpe?g)$/))

export { testData }

/*
// Helper func to check where numbering of images is off -- to be run within index of folder 
const importAll = req => {
  let images = {}
  req.keys().map(item => {
    images[item.replace('./long-sleeve-', '').replace(/\.(png|jpe?g)$/, '')] = req(item)
  })
  let count = 1
  for (const image in images) {
    console.log(image)
    if (+image !== count) {
      console.log('here')
      break;
    }
    count++
  }
  return images
}
*/