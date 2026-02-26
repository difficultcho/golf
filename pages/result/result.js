// result.js
const config = require('../../utils/config.js')

Page({
  data: {
    videoId: '',
    videoUrl: '',
    metadata: null,
    rawMetrics: null,
    suggestions: [],
    videoLoading: true,
    videoError: '',
    activeTab: 0,
    chartsDrawn: false
  },

  onLoad(options) {
    const videoId = options.videoId
    if (!videoId) {
      wx.showToast({
        title: '视频ID无效',
        icon: 'none'
      })
      setTimeout(() => {
        wx.navigateBack()
      }, 1500)
      return
    }

    console.log('Video ID:', videoId)

    this.setData({ videoId })

    // Download video with auth header, then set local path as src
    this.loadVideo(videoId)
    this.loadMetadata()
  },

  loadVideo(videoId) {
    const remoteUrl = `${config.API_BASE}/api/video/result/${videoId}`
    wx.downloadFile({
      url: remoteUrl,
      header: config.authHeader(),
      success: (res) => {
        if (res.statusCode === 200) {
          this.setData({ videoUrl: res.tempFilePath })
        } else {
          console.error('Video download failed:', res.statusCode)
          this.setData({ videoError: '视频加载失败', videoLoading: false })
        }
      },
      fail: (err) => {
        console.error('Video download error:', err)
        this.setData({ videoError: '视频下载失败', videoLoading: false })
      }
    })
  },

  onVideoLoaded() {
    console.log('Video loaded successfully')
    this.setData({
      videoLoading: false,
      videoError: ''
    })
  },

  onVideoError(e) {
    console.error('Video error:', e)
    const errorMsg = e.detail && e.detail.errMsg
      ? e.detail.errMsg
      : '视频加载失败'

    this.setData({
      videoLoading: false,
      videoError: errorMsg
    })

    wx.showToast({
      title: '视频加载失败',
      icon: 'none',
      duration: 3000
    })
  },

  loadMetadata() {
    wx.request({
      url: `${config.API_BASE}/api/video/data/${this.data.videoId}`,
      method: 'GET',
      header: config.authHeader(),
      success: (res) => {
        console.log('Metadata response:', res)

        if (res.statusCode === 200) {
          const data = res.data
          let rawMetrics = null

          // Extract raw numeric values for charts BEFORE formatting
          if (data.club_head_speed_mph != null) {
            rawMetrics = {
              clubHeadSpeed: parseFloat(data.club_head_speed_mph),
              xFactor: parseFloat(data.x_factor),
              balanceScore: parseFloat(data.balance_score),
              energyEfficiency: parseFloat(data.energy_efficiency) * 100,
              swingDuration: parseFloat(data.swing_duration_sec),
              peakTorques: {
                shoulder: parseFloat(data.peak_torques.shoulder),
                hip: parseFloat(data.peak_torques.hip),
                knee: parseFloat(data.peak_torques.knee)
              }
            }

            // Format for display
            data.club_head_speed_mph = parseFloat(data.club_head_speed_mph).toFixed(1)
            data.x_factor = parseFloat(data.x_factor).toFixed(1)
            data.balance_score = parseFloat(data.balance_score).toFixed(0)
            data.energy_efficiency_pct = (parseFloat(data.energy_efficiency) * 100).toFixed(1)
            data.swing_duration_sec = parseFloat(data.swing_duration_sec).toFixed(2)
            if (data.peak_torques) {
              data.peak_torques.shoulder = parseFloat(data.peak_torques.shoulder).toFixed(1)
              data.peak_torques.hip = parseFloat(data.peak_torques.hip).toFixed(1)
              data.peak_torques.knee = parseFloat(data.peak_torques.knee).toFixed(1)
            }
          }
          if (data.duration != null) {
            data.duration = parseFloat(data.duration).toFixed(2)
          }
          if (data.processing_time != null) {
            data.processing_time = parseFloat(data.processing_time).toFixed(2)
          }
          const suggestions = this.generateSuggestions(rawMetrics)
          this.setData({ metadata: data, rawMetrics, suggestions })
          this.saveToHistory(videoId, rawMetrics)
        } else {
          wx.showToast({
            title: '元数据加载失败',
            icon: 'none'
          })
        }
      },
      fail: (err) => {
        console.error('Load metadata failed:', err)
        wx.showToast({
          title: '网络错误',
          icon: 'none'
        })
      }
    })
  },

  switchTab(e) {
    const tab = parseInt(e.currentTarget.dataset.tab, 10)
    const wasCharts = this.data.activeTab === 1
    this.setData({
      activeTab: tab,
      chartsDrawn: wasCharts ? false : this.data.chartsDrawn
    })

    if (tab === 1 && this.data.rawMetrics) {
      wx.nextTick(() => {
        this.drawCharts()
      })
    }
  },

  drawCharts() {
    this.drawRadarChart()
    this.drawBarChart()
    this.setData({ chartsDrawn: true })
  },

  drawRadarChart() {
    const query = wx.createSelectorQuery()
    query.select('#radarCanvas')
      .fields({ node: true, size: true })
      .exec((res) => {
        if (!res[0]) return
        const canvas = res[0].node
        const ctx = canvas.getContext('2d')
        const dpr = wx.getWindowInfo().pixelRatio

        canvas.width = res[0].width * dpr
        canvas.height = res[0].height * dpr
        ctx.scale(dpr, dpr)

        const width = res[0].width
        const height = res[0].height
        const centerX = width / 2
        const centerY = height / 2
        const radius = Math.min(width, height) / 2 - 36

        const raw = this.data.rawMetrics
        const metrics = [
          { label: '杆头速度', value: Math.min(raw.clubHeadSpeed / 120, 1) },
          { label: 'X-Factor', value: Math.min(raw.xFactor / 50, 1) },
          { label: '平衡得分', value: Math.min(raw.balanceScore / 100, 1) },
          { label: '能量效率', value: Math.min(raw.energyEfficiency / 100, 1) }
        ]
        const numAxes = metrics.length
        const angleStep = (2 * Math.PI) / numAxes
        const startAngle = -Math.PI / 2

        // Draw 3 concentric rings
        ctx.strokeStyle = '#e0e0e0'
        ctx.lineWidth = 1
        for (let ring = 1; ring <= 3; ring++) {
          const r = (radius * ring) / 3
          ctx.beginPath()
          for (let i = 0; i <= numAxes; i++) {
            const angle = startAngle + i * angleStep
            const x = centerX + r * Math.cos(angle)
            const y = centerY + r * Math.sin(angle)
            if (i === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          }
          ctx.closePath()
          ctx.stroke()
        }

        // Draw axis lines
        for (let i = 0; i < numAxes; i++) {
          const angle = startAngle + i * angleStep
          ctx.beginPath()
          ctx.moveTo(centerX, centerY)
          ctx.lineTo(centerX + radius * Math.cos(angle), centerY + radius * Math.sin(angle))
          ctx.stroke()
        }

        // Draw filled data polygon
        ctx.beginPath()
        ctx.fillStyle = 'rgba(7, 193, 96, 0.25)'
        ctx.strokeStyle = '#07c160'
        ctx.lineWidth = 2
        for (let i = 0; i < numAxes; i++) {
          const angle = startAngle + i * angleStep
          const r = radius * metrics[i].value
          const x = centerX + r * Math.cos(angle)
          const y = centerY + r * Math.sin(angle)
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.closePath()
        ctx.fill()
        ctx.stroke()

        // Draw data points
        for (let i = 0; i < numAxes; i++) {
          const angle = startAngle + i * angleStep
          const r = radius * metrics[i].value
          const x = centerX + r * Math.cos(angle)
          const y = centerY + r * Math.sin(angle)
          ctx.beginPath()
          ctx.arc(x, y, 4, 0, 2 * Math.PI)
          ctx.fillStyle = '#07c160'
          ctx.fill()
        }

        // Draw labels
        ctx.fillStyle = '#333'
        ctx.font = '12px sans-serif'
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        for (let i = 0; i < numAxes; i++) {
          const angle = startAngle + i * angleStep
          const labelRadius = radius + 22
          const x = centerX + labelRadius * Math.cos(angle)
          const y = centerY + labelRadius * Math.sin(angle)
          ctx.fillText(metrics[i].label, x, y)
        }
      })
  },

  drawBarChart() {
    const query = wx.createSelectorQuery()
    query.select('#barCanvas')
      .fields({ node: true, size: true })
      .exec((res) => {
        if (!res[0]) return
        const canvas = res[0].node
        const ctx = canvas.getContext('2d')
        const dpr = wx.getWindowInfo().pixelRatio

        canvas.width = res[0].width * dpr
        canvas.height = res[0].height * dpr
        ctx.scale(dpr, dpr)

        const width = res[0].width
        const height = res[0].height
        const raw = this.data.rawMetrics

        const torques = [
          { label: '肩关节', value: raw.peakTorques.shoulder, color: '#07c160' },
          { label: '髋关节', value: raw.peakTorques.hip, color: '#1989fa' },
          { label: '膝关节', value: raw.peakTorques.knee, color: '#ff976a' }
        ]

        const maxVal = Math.max(...torques.map(t => t.value)) * 1.2
        const paddingLeft = 60
        const paddingRight = 80
        const barAreaWidth = width - paddingLeft - paddingRight
        const barHeight = 28
        const barGap = 24
        const totalHeight = torques.length * (barHeight + barGap) - barGap
        const startY = (height - totalHeight) / 2

        torques.forEach((t, i) => {
          const y = startY + i * (barHeight + barGap)
          const barWidth = (t.value / maxVal) * barAreaWidth

          // Background bar
          ctx.fillStyle = '#f0f0f0'
          ctx.fillRect(paddingLeft, y, barAreaWidth, barHeight)

          // Value bar with rounded ends
          ctx.fillStyle = t.color
          if (ctx.roundRect) {
            ctx.beginPath()
            ctx.roundRect(paddingLeft, y, barWidth, barHeight, 4)
            ctx.fill()
          } else {
            ctx.fillRect(paddingLeft, y, barWidth, barHeight)
          }

          // Label (left)
          ctx.fillStyle = '#666'
          ctx.font = '12px sans-serif'
          ctx.textAlign = 'right'
          ctx.textBaseline = 'middle'
          ctx.fillText(t.label, paddingLeft - 8, y + barHeight / 2)

          // Value (right of bar)
          ctx.fillStyle = '#333'
          ctx.font = '11px sans-serif'
          ctx.textAlign = 'left'
          ctx.fillText(`${t.value.toFixed(1)} Nm`, paddingLeft + barWidth + 6, y + barHeight / 2)
        })
      })
  },

  generateSuggestions(raw) {
    if (!raw) return []
    const tips = []
    if (raw.clubHeadSpeed < 80) {
      tips.push({ level: 'warn', text: '杆头速度偏低，建议加强核心旋转训练与下肢蹬地发力练习' })
    } else {
      tips.push({ level: 'good', text: `杆头速度 ${raw.clubHeadSpeed.toFixed(1)} mph，表现良好` })
    }
    if (raw.xFactor < 30) {
      tips.push({ level: 'warn', text: 'X-Factor 偏小，上杆时尝试增大肩部转动、减少髋部旋转' })
    } else {
      tips.push({ level: 'good', text: `X-Factor ${raw.xFactor.toFixed(1)}°，肩髋分离度理想` })
    }
    if (raw.balanceScore < 70) {
      tips.push({ level: 'warn', text: '平衡稳定性待提升，注意保持下盘稳固，控制重心转移节奏' })
    } else {
      tips.push({ level: 'good', text: `平衡得分 ${raw.balanceScore.toFixed(0)} 分，重心控制稳定` })
    }
    if (raw.energyEfficiency < 65) {
      tips.push({ level: 'warn', text: '动力链传递效率偏低，优化腿→躯干→手臂的顺序发力顺序' })
    } else {
      tips.push({ level: 'good', text: `能量传递效率 ${raw.energyEfficiency.toFixed(1)}%，动力链流畅` })
    }
    return tips
  },

  saveToHistory(videoId, rawMetrics) {
    try {
      const history = wx.getStorageSync('swingHistory') || []
      if (history.find(h => h.id === videoId)) return
      const now = new Date()
      const dateFormatted = `${now.getMonth() + 1}月${now.getDate()}日 ${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`
      history.unshift({
        id: videoId,
        date: now.toISOString(),
        dateFormatted,
        metrics: rawMetrics ? {
          clubHeadSpeed: rawMetrics.clubHeadSpeed.toFixed(1),
          balanceScore: rawMetrics.balanceScore.toFixed(0)
        } : null
      })
      if (history.length > 20) history.pop()
      wx.setStorageSync('swingHistory', history)
    } catch (e) {
      console.error('Failed to save history:', e)
    }
  },

  downloadVideo() {
    wx.showLoading({
      title: '下载中...'
    })

    wx.downloadFile({
      url: this.data.videoUrl,
      header: config.authHeader(),
      success: (res) => {
        if (res.statusCode === 200) {
          wx.getSetting({
            success: (settingRes) => {
              if (settingRes.authSetting['scope.writePhotosAlbum']) {
                this.saveToAlbum(res.tempFilePath)
              } else {
                wx.authorize({
                  scope: 'scope.writePhotosAlbum',
                  success: () => {
                    this.saveToAlbum(res.tempFilePath)
                  },
                  fail: () => {
                    wx.hideLoading()
                    wx.showModal({
                      title: '需要相册权限',
                      content: '请在设置中开启相册权限',
                      confirmText: '去设置',
                      success: (modalRes) => {
                        if (modalRes.confirm) {
                          wx.openSetting()
                        }
                      }
                    })
                  }
                })
              }
            }
          })
        } else {
          wx.hideLoading()
          wx.showToast({
            title: '下载失败',
            icon: 'none'
          })
        }
      },
      fail: (err) => {
        wx.hideLoading()
        console.error('Download failed:', err)
        wx.showToast({
          title: '下载失败',
          icon: 'none'
        })
      }
    })
  },

  saveToAlbum(filePath) {
    wx.saveVideoToPhotosAlbum({
      filePath: filePath,
      success: () => {
        wx.hideLoading()
        wx.showToast({
          title: '已保存到相册',
          icon: 'success'
        })
      },
      fail: (err) => {
        wx.hideLoading()
        console.error('Save to album failed:', err)
        wx.showToast({
          title: '保存失败',
          icon: 'none'
        })
      }
    })
  },

  downloadData() {
    if (!this.data.metadata) {
      wx.showToast({
        title: '元数据未加载',
        icon: 'none'
      })
      return
    }

    const metadataStr = JSON.stringify(this.data.metadata, null, 2)

    wx.showModal({
      title: '视频数据',
      content: metadataStr,
      showCancel: true,
      cancelText: '关闭',
      confirmText: '复制',
      success: (res) => {
        if (res.confirm) {
          wx.setClipboardData({
            data: metadataStr,
            success: () => {
              wx.showToast({
                title: '已复制到剪贴板',
                icon: 'success'
              })
            }
          })
        }
      }
    })
  },

  backToHome() {
    wx.reLaunch({
      url: '/pages/index/index'
    })
  }
})
