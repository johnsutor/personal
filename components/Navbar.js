import Link from 'next/link'

export default function NavBar() {
  return (
    <div className="flex justify-between py-4">
        <Link href="/">
            <a>
            <h1 className="text-2xl font-bold w-1/2 cursor-pointer">
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
            <a href="https://drive.google.com/file/d/1co1kKJcJeSdXAfNMzW4-JsUQoxNXYvcP/view?usp=sharing" target="_blank" classNamepx-2>
                CV
            </a>
        </div>
    </div>
  )
}