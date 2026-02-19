// record.js
Page({
  data: {
    videoPath: '',
    recording: false,
    recordTime: 0,
    maxRecordTime: 60, // Maximum 60 seconds
    timer: null
  },

  onLoad() {
    this.ctx = wx.createCameraContext()
  },

  startRecord() {
    // Request camera permission if needed
    wx.authorize({
      scope: 'scope.camera',
      success: () => {
        this.doStartRecord()
      },
      fail: () => {
        wx.showModal({
          title: '需要相机权限',
          content: '请在设置中开启相机权限',
          confirmText: '去设置',
          success: (res) => {
            if (res.confirm) {
              wx.openSetting()
            }
          }
        })
      }
    })
  },

  doStartRecord() {
    this.setData({ recording: true, recordTime: 0 })

    // Start recording
    this.ctx.startRecord({
      success: () => {
        console.log('Recording started')
        this.startTimer()
      },
      fail: (err) => {
        console.error('Recording failed:', err)
        wx.showToast({
          title: '录制失败',
          icon: 'none'
        })
        this.setData({ recording: false })
      }
    })
  },

  stopRecord() {
    if (this.data.timer) {
      clearInterval(this.data.timer)
    }

    this.ctx.stopRecord({
      success: (res) => {
        console.log('Recording stopped:', res.tempVideoPath)
        this.setData({
          videoPath: res.tempVideoPath,
          recording: false
        })
      },
      fail: (err) => {
        console.error('Stop recording failed:', err)
        wx.showToast({
          title: '停止录制失败',
          icon: 'none'
        })
        this.setData({ recording: false })
      }
    })
  },

  startTimer() {
    const timer = setInterval(() => {
      const time = this.data.recordTime + 1
      this.setData({ recordTime: time })

      // Auto stop when reaching max time
      if (time >= this.data.maxRecordTime) {
        this.stopRecord()
      }
    }, 1000)

    this.setData({ timer })
  },

  retake() {
    this.setData({
      videoPath: '',
      recordTime: 0
    })
  },

  uploadVideo() {
    if (!this.data.videoPath) {
      wx.showToast({
        title: '请先录制视频',
        icon: 'none'
      })
      return
    }

    wx.navigateTo({
      url: '/pages/upload/upload?videoPath=' + encodeURIComponent(this.data.videoPath)
    })
  },

  onUnload() {
    // Clean up timer when leaving page
    if (this.data.timer) {
      clearInterval(this.data.timer)
    }
  }
})
