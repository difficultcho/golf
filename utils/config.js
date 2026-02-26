// API configuration for the backend server
// Change this for production deployment

const ENV = 'development'  // 'development' | 'production'

const config = {
  development: {
    API_BASE: 'http://127.0.0.1:8000'  // Must match backend port (config.py PORT)
  },
  production: {
    API_BASE: 'https://bce.kkmsee.com'
  }
}

const current = config[ENV]

/**
 * Get Authorization header with current token.
 * Usage: header: config.authHeader()
 */
current.authHeader = function () {
  const app = getApp()
  const token = app ? app.getToken() : ''
  return token ? { Authorization: `Bearer ${token}` } : {}
}

module.exports = current
