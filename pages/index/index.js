// index.js
Page({
  data: {},

  navigateToRecord() {
    wx.navigateTo({
      url: '/pages/record/record'
    })
  },

  chooseLocalVideo() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['video'],
      sourceType: ['album'],
      maxDuration: 60,
      success: (res) => {
        const videoPath = res.tempFiles[0].tempFilePath
        console.log('Selected local video:', videoPath)
        wx.navigateTo({
          url: '/pages/upload/upload?videoPath=' + encodeURIComponent(videoPath)
        })
      },
      fail: (err) => {
        console.log('Video selection cancelled or failed:', err)
      }
    })
  }
})
