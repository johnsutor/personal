import Link from 'next/link'

export default function NavBar() {
    return (
        <div className="flex justify-between py-4">
            <Link href="/">
                <h1 className="text-2xl font-bold cursor-pointer">
                    John Sutor
                </h1>
            </Link>
            <div className="flex justify-end w-1/2">
                <Link href="/" className="px-2 cursor-pointer text-gray-700 transition-transform hover:text-black hover:scale-110">
                    Home
                </Link>
                <Link href="/press" className="px-2 cursor-pointer text-gray-700 transition-transform hover:text-black hover:scale-110">
                    Press
                </Link>
                <Link href="/resume_john_sutor.pdf" target="_blank" className="px-2 text-gray-700 transition-transform hover:text-black hover:scale-110">
                    Resume
                </Link>
            </div>
        </div >
    )
}