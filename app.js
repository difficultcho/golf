// app.js
const config = require('./utils/config.js')

App({
  globalData: {
    token: '',
    userInfo: null,
    _loginCallbacks: []  // Pages waiting for login result
  },

  onLaunch() {
    // Only restore cached token, don't auto-login
    const token = wx.getStorageSync('token')
    if (token) {
      this.globalData.token = token
      this.fetchUserInfo()
    }
  },

  /**
   * Trigger wx.login → backend auth → store token.
   * Returns result via callback: cb(success: boolean)
   */
  login(cb) {
    if (cb) this.globalData._loginCallbacks.push(cb)

    wx.login({
      success: (res) => {
        if (!res.code) {
          console.error('wx.login failed: no code')
          this._notifyLogin(false)
          return
        }
        wx.request({
          url: `${config.API_BASE}/api/auth/login`,
          method: 'POST',
          data: { code: res.code },
          success: (resp) => {
            if (resp.statusCode === 200) {
              const { token, user_id, nickname, role, is_new_user } = resp.data
              this.globalData.token = token
              this.globalData.userInfo = { id: user_id, nickname, role }
              wx.setStorageSync('token', token)
              console.log('Login success:', is_new_user ? 'new user' : 'existing user')
              this._notifyLogin(true)
            } else {
              console.error('Login API failed:', resp.statusCode, resp.data)
              this._notifyLogin(false)
            }
          },
          fail: (err) => {
            console.error('Login request failed:', err)
            this._notifyLogin(false)
          }
        })
      },
      fail: () => {
        this._notifyLogin(false)
      }
    })
  },

  _notifyLogin(success) {
    const cbs = this.globalData._loginCallbacks
    this.globalData._loginCallbacks = []
    cbs.forEach(cb => cb(success))
  },

  fetchUserInfo() {
    wx.request({
      url: `${config.API_BASE}/api/auth/me`,
      method: 'GET',
      header: { Authorization: `Bearer ${this.globalData.token}` },
      success: (res) => {
        if (res.statusCode === 200) {
          this.globalData.userInfo = res.data
        } else if (res.statusCode === 401) {
          // Token expired, clear it
          this.logout()
        }
      }
    })
  },

  isLoggedIn() {
    return !!this.globalData.token
  },

  getToken() {
    return this.globalData.token
  },

  logout() {
    this.globalData.token = ''
    this.globalData.userInfo = null
    wx.removeStorageSync('token')
  }
})
