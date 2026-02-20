// index.js
Page({
  data: {
    activeTab: 0,
    history: [],
    statusBarHeight: 44
  },

  onLoad() {
    const info = wx.getWindowInfo()
    this.setData({ statusBarHeight: info.statusBarHeight })
  },

  onShow() {
    this.loadHistory()
  },

  loadHistory() {
    try {
      const history = wx.getStorageSync('swingHistory') || []
      this.setData({ history })
    } catch (e) {
      this.setData({ history: [] })
    }
  },

  switchTab(e) {
    const tab = parseInt(e.currentTarget.dataset.tab, 10)
    this.setData({ activeTab: tab })
  },

  startRecord() {
    wx.navigateTo({ url: '/pages/record/record' })
  },

  chooseAlbum() {
    wx.chooseMedia({
      count: 1,
      mediaType: ['video'],
      sourceType: ['album'],
      maxDuration: 60,
      success: (res) => {
        const videoPath = res.tempFiles[0].tempFilePath
        wx.navigateTo({
          url: '/pages/upload/upload?videoPath=' + encodeURIComponent(videoPath)
        })
      },
      fail: () => {}
    })
  },

  viewResult(e) {
    const id = e.currentTarget.dataset.id
    wx.navigateTo({ url: '/pages/result/result?videoId=' + id })
  },

  clearHistory() {
    wx.showModal({
      title: '清除记录',
      content: '确定要清除所有挥杆记录吗？此操作不可恢复。',
      confirmText: '清除',
      confirmColor: '#e74c3c',
      success: (res) => {
        if (res.confirm) {
          wx.removeStorageSync('swingHistory')
          this.setData({ history: [], activeTab: 0 })
          wx.showToast({ title: '已清除', icon: 'success' })
        }
      }
    })
  },

  showAbout() {
    wx.showModal({
      title: 'Golf AI',
      content: 'v1.0.0\n\nAI 驱动的高尔夫挥杆分析系统，结合计算机视觉与生物力学模型，为每位球手提供专属改进建议。',
      showCancel: false,
      confirmText: '知道了'
    })
  },

  sendFeedback() {
    wx.showToast({ title: '敬请期待', icon: 'none' })
  }
})
