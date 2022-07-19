import Link from 'next/link'

export default function NavBar() {
  return (
    <div className="flex justify-between py-4">
        <Link href="/">
            <a>
            <h1 className="text-2xl font-bold cursor-pointer">
                John Sutor
            </h1>
            </a>
        </Link>
        <div className="flex justify-end w-1/2">
            <Link href="/">
                <a className="px-2 cursor-pointer text-gray-700 transition-transform hover:text-black hover:scale-110">
                    Home
                </a>
            </Link>
            <Link href="/press">
                <a className="px-2 cursor-pointer text-gray-700 transition-transform hover:text-black hover:scale-110">
                    Press
                </a>
            </Link>
            <a href="/resume_js.pdf" target="_blank" className="px-2 text-gray-700 transition-transform hover:text-black hover:scale-110">
                Resume
            </a>
        </div>
    </div>
  )
}