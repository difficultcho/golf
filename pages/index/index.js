// index.js
Page({
  data: {
    activeTab: 0,
    history: [],
    statusBarHeight: 44,
    loggedIn: false,
    nickname: '',
    role: '',
    avatarUrl: ''
  },

  onLoad() {
    const info = wx.getWindowInfo()
    this.setData({ statusBarHeight: info.statusBarHeight })
  },

  onShow() {
    const app = getApp()
    const loggedIn = app.isLoggedIn()
    this.setData({ loggedIn })

    if (loggedIn) {
      this.loadHistory()
      this.loadUserInfo()
    }
  },

  loadUserInfo() {
    const app = getApp()
    const info = app.globalData.userInfo
    if (info) {
      this.setData({
        nickname: info.nickname,
        role: info.role === 'coach' ? '教练' : info.role === 'admin' ? '管理员' : '',
        avatarUrl: info.avatar_url || ''
      })
    }
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

  doLogin() {
    const app = getApp()
    wx.showLoading({ title: '登录中...' })
    app.login((success) => {
      wx.hideLoading()
      if (success) {
        this.setData({ loggedIn: true })
        this.loadHistory()
        this.loadUserInfo()
        wx.showToast({ title: '登录成功', icon: 'success' })
      } else {
        wx.showToast({ title: '登录失败，请重试', icon: 'none' })
      }
    })
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
  },

  logout() {
    wx.showModal({
      title: '退出登录',
      content: '退出后需要重新使用微信身份登录，确定退出？',
      confirmText: '退出',
      confirmColor: '#e74c3c',
      success: (res) => {
        if (res.confirm) {
          const app = getApp()
          app.logout()
          this.setData({ loggedIn: false, nickname: '', role: '', avatarUrl: '', activeTab: 0 })
          wx.showToast({ title: '已退出', icon: 'success' })
        }
      }
    })
  }
})
