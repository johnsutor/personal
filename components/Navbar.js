import Link from 'next/link'

export default function NavBar() {
  return (
    <div className="flex justify-between py-4">
        <Link href="/">
            <h1 className="text-2xl font-bold w-1/2 cursor-pointer">
                John Sutor
            </h1>
        </Link>
        <div className="flex justify-end w-1/2">
            <Link href="/">
                <span className="px-2 cursor-pointer">
                    Home
                </span>
            </Link>
            <Link href="/papers">
                <span className="px-2 cursor-pointer">
                    Papers
                </span>
            </Link>
            <Link href="/press">
                <span className="px-2 cursor-pointer">
                    Press
                </span>
            </Link>
        </div>
    </div>
  )
}