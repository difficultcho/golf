// upload.js
const config = require('../../utils/config.js')

Page({
  data: {
    videoPath: '',
    uploadProgress: 0,
    statusText: '准备上传...',
    videoId: '',
    success: false,
    failed: false,
    step: 0  // 0=uploading, 1=processing, 2=done, -1=error
  },

  onLoad(options) {
    console.log('=== Upload page onLoad ===')
    console.log('Options:', options)

    const videoPath = decodeURIComponent(options.videoPath || '')
    console.log('Decoded videoPath:', videoPath)

    if (!videoPath) {
      console.error('Invalid videoPath, navigating back')
      wx.showToast({
        title: '视频路径无效',
        icon: 'none'
      })
      setTimeout(() => {
        wx.navigateBack()
      }, 1500)
      return
    }

    this.setData({ videoPath })
    console.log('Starting upload...')
    // Start upload automatically
    this.uploadVideo()
  },

  uploadVideo() {
    console.log('=== uploadVideo called ===')
    console.log('Video path:', this.data.videoPath)
    console.log('API URL:', `${config.API_BASE}/api/video/upload`)

    this.setData({
      uploadProgress: 0,
      statusText: '正在上传...',
      failed: false,
      success: false,
      step: 0
    })

    const uploadTask = wx.uploadFile({
      url: `${config.API_BASE}/api/video/upload`,
      filePath: this.data.videoPath,
      name: 'video',
      success: (res) => {
        console.log('=== Upload SUCCESS ===')
        console.log('Status code:', res.statusCode)
        console.log('Response data:', res.data)
        console.log('Response data type:', typeof res.data)

        if (res.statusCode === 200) {
          try {
            const data = JSON.parse(res.data)
            console.log('Parsed data:', data)
            console.log('Video ID:', data.video_id)

            this.setData({
              videoId: data.video_id,
              statusText: 'AI 正在分析挥杆动作...',
              uploadProgress: 100,
              step: 1
            })
            console.log('Starting pollStatus...')
            // Start polling status
            this.pollStatus()
          } catch (e) {
            console.error('=== Parse response error ===', e)
            console.error('Raw data:', res.data)
            this.handleError('响应解析失败')
          }
        } else {
          console.error('=== Upload failed with status ===', res.statusCode)
          console.error('Response:', res)
          this.handleError(`上传失败 (${res.statusCode})`)
        }
      },
      fail: (err) => {
        console.error('=== Upload FAILED ===', err)
        this.handleError('网络错误，请检查连接')
      }
    })

    // Update progress
    uploadTask.onProgressUpdate((res) => {
      console.log('Upload progress:', res.progress)
      this.setData({
        uploadProgress: res.progress,
        statusText: `上传中 ${res.progress}%`
      })
    })
  },

  pollStatus() {
    console.log('=== pollStatus called ===')
    console.log('Video ID:', this.data.videoId)

    const checkStatus = () => {
      console.log('=== Checking status ===')
      const statusUrl = `${config.API_BASE}/api/video/status/${this.data.videoId}`
      console.log('Status URL:', statusUrl)

      wx.request({
        url: statusUrl,
        method: 'GET',
        success: (res) => {
          console.log('=== Status response ===')
          console.log('Status code:', res.statusCode)
          console.log('Response data:', res.data)

          if (res.statusCode === 200) {
            const status = res.data.status
            console.log('Current status:', status)

            if (status === 'done') {
              console.log('=== Processing DONE ===')
              this.setData({
                success: true,
                statusText: '分析完成！',
                step: 2
              })
              console.log('Success state set, data:', this.data)
            } else if (status === 'failed') {
              console.error('=== Processing FAILED ===')
              this.handleError('视频处理失败')
            } else {
              console.log('=== Still processing, will poll again ===')
              // Still processing, poll again
              this.setData({
                statusText: '正在分析挥杆动作，请稍候...'
              })
              setTimeout(checkStatus, 3000) // Check every 3 seconds
            }
          } else {
            console.error('=== Status check returned non-200 ===', res.statusCode)
            this.handleError('状态查询失败')
          }
        },
        fail: (err) => {
          console.error('=== Status check FAILED ===', err)
          this.handleError('网络错误')
        }
      })
    }

    // Start polling (delay first check so user can see "upload complete" message)
    console.log('Starting first status check in 2s...')
    setTimeout(checkStatus, 2000)
  },

  handleError(message) {
    console.error('=== handleError called ===')
    console.error('Error message:', message)
    this.setData({
      failed: true,
      statusText: message,
      step: -1
    })
    wx.showToast({
      title: message,
      icon: 'none',
      duration: 2000
    })
  },

  retryUpload() {
    this.uploadVideo()
  },

  viewResult() {
    console.log('=== viewResult called ===')
    console.log('Video ID:', this.data.videoId)

    if (!this.data.videoId) {
      console.error('Video ID is invalid')
      wx.showToast({
        title: '视频ID无效',
        icon: 'none'
      })
      return
    }

    const resultUrl = '/pages/result/result?videoId=' + this.data.videoId
    console.log('Navigating to:', resultUrl)

    wx.redirectTo({
      url: resultUrl
    })
  }
})
