// components/layout.js

import NavBar from './Navbar'
import Head from 'next/head'

export default function Layout({ children }) {
  return (
    <div className="min-h-screen px-2 md:w-2/3 mx-auto">
        <Head>
          <title>John Sutor's Website</title>
          <link rel="icon" href="/favicon.ico" />
        </Head>
      <NavBar />
      <main>{children}</main>
    </div>
  )
}