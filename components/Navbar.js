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
                <a>
                <span className="px-2 cursor-pointer">
                    Home
                </span>
                </a>
            </Link>
            <Link href="/papers">
                <a>
                <span className="px-2 cursor-pointer">
                    Papers
                </span>
                </a>
            </Link>
            <Link href="/press">
                <a>
                <span className="px-2 cursor-pointer">
                    Press
                </span>
                </a>
            </Link>
            <a href="/sutor_cv.pdf" target="_blank" className="px-2">
                CV
            </a>
        </div>
    </div>
  )
}