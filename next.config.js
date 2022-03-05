module.exports = {
    webpack: function(config) {
      config.module.rules.push({
        test: /\.md$/,
        use: 'raw-loader',
      })
      return config
    },
    async headers() {
      return [
          {
              source: '/:path*',
              headers: [
                  {
                      key: 'Strict-Transport-Security',
                      value: 'max-age=63072000; includeSubDomains; preload',
                  },
                  {
                      key: 'X-Frame-Options',
                      value: 'SAMEORIGIN',
                  },
                  {
                      key: 'X-XSS-Protection',
                      value: '1; mode=block'
                  },
                  {
                      key: 'X-Content-Type-Options',
                      value: 'nosniff'
                  },
                  {
                      key: 'Referrer-Policy',
                      value: 'origin-when-cross-origin',
                  },
                  {
                      key: 'Permissions-Policy',
                      value: 'accelerometer=(self), ambient-light-sensor=(self), autoplay=(self), battery=(self), camera=(self), cross-origin-isolated=(self), display-capture=(self), document-domain=(self), encrypted-media=(self), execution-while-not-rendered=(self), execution-while-out-of-viewport=(self), fullscreen=(self), geolocation=(self), gyroscope=(self), keyboard-map=(self), magnetometer=(self), microphone=(self), midi=(self), navigation-override=(self), payment=(self), picture-in-picture=(self), publickey-credentials-get=(self), screen-wake-lock=(self), sync-xhr=(self), usb=(self), web-share=(self), xr-spatial-tracking=(self)'
                  },
                  {
                      key: 'X-DNS-Prefetch-Control',
                      value: 'on'
                  }
              ],
          },
      ]
    },
  }