import math from 'remark-math'
import katex from 'rehype-katex'
import nextMDX from '@next/mdx'

/** @type {import('next').NextConfig} */

const nextConfig = {
  reactStrictMode: true,
  pageExtensions: ['js', 'jsx', 'md', 'mdx', 'ts', 'tsx'],
  async headers() {
    return [
      {
        source: '/:path*',
        headers: [
          {
            key: 'Strict-Transport-Security',
            value:
              'max-age=63072000; includeSubDomains; preload',
          },
          {
            key: 'X-Frame-Options',
            value: 'SAMEORIGIN',
          },
          {
            key: 'X-XSS-Protection',
            value: '1; mode=block',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Content-Security-Policy',
            value: "default-src 'self' johnsutor.com *.johnsutor.com googleapis.com gstatic.com googleusercontent.com;  img-src 'self' artsandsciences.fsu.edu www.neonscience.org pbs.twimg.com s3-us-west-2.amazonaws.com news.cci.fsu.edu news.fsu.edu;  script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline';  manifest-src 'self'; base-uri 'self'; frame-ancestors 'none'; form-action 'none';"
          },
          {
            key: 'Referrer-Policy',
            value: 'origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value:
              'accelerometer=(self), ambient-light-sensor=(self), autoplay=(self), battery=(self), camera=(self), cross-origin-isolated=(self), display-capture=(self), document-domain=(self), encrypted-media=(self), execution-while-not-rendered=(self), execution-while-out-of-viewport=(self), fullscreen=(self), geolocation=(self), gyroscope=(self), keyboard-map=(self), magnetometer=(self), microphone=(self), midi=(self), navigation-override=(self), payment=(self), picture-in-picture=(self), publickey-credentials-get=(self), screen-wake-lock=(self), sync-xhr=(self), usb=(self), web-share=(self), xr-spatial-tracking=(self)',
          },
          {
            key: 'X-DNS-Prefetch-Control',
            value: 'on',
          },
        ],
      },
    ]
  },
}

const withMDX = nextMDX({
  extension: /\.mdx?$/,
  options: {
    providerImportSource: '@mdx-js/react',
    remarkPlugins: [math],
    rehypePlugins: [katex],
  },
})

export default withMDX({
  ...nextConfig,
  pageExtensions: ['js', 'jsx', 'md', 'mdx', 'ts', 'tsx'],
})