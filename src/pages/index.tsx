import Head from 'next/head'
import Image from 'next/image'
import { useState } from 'react';
import Layout from '@/components/Layout';
import Link from 'next/link';

export default function Home() {
  const [profiles, setProfiles] = useState([
    {
      name: "GitHub",
      url: "https://github.com/johnsutor",
      icon:
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
            <path d="M256 32C132.3 32 32 134.9 32 261.7c0 101.5 64.2 187.5 153.2 217.9 1.4.3 2.6.4 3.8.4 8.3 0 11.5-6.1 11.5-11.4 0-5.5-.2-19.9-.3-39.1-8.4 1.9-15.9 2.7-22.6 2.7-43.1 0-52.9-33.5-52.9-33.5-10.2-26.5-24.9-33.6-24.9-33.6-19.5-13.7-.1-14.1 1.4-14.1h.1c22.5 2 34.3 23.8 34.3 23.8 11.2 19.6 26.2 25.1 39.6 25.1 10.5 0 20-3.4 25.6-6 2-14.8 7.8-24.9 14.2-30.7-49.7-5.8-102-25.5-102-113.5 0-25.1 8.7-45.6 23-61.6-2.3-5.8-10-29.2 2.2-60.8 0 0 1.6-.5 5-.5 8.1 0 26.4 3.1 56.6 24.1 17.9-5.1 37-7.6 56.1-7.7 19 .1 38.2 2.6 56.1 7.7 30.2-21 48.5-24.1 56.6-24.1 3.4 0 5 .5 5 .5 12.2 31.6 4.5 55 2.2 60.8 14.3 16.1 23 36.6 23 61.6 0 88.2-52.4 107.6-102.3 113.3 8 7.1 15.2 21.1 15.2 42.5 0 30.7-.3 55.5-.3 63 0 5.4 3.1 11.5 11.4 11.5 1.2 0 2.6-.1 4-.4C415.9 449.2 480 363.1 480 261.7 480 134.9 379.7 32 256 32z"/>
        </svg>`
    },
    {
      name: "Instagram",
      url: "https://www.instagram.com/john_sutor/",
      icon:
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
            <path d="M336 96c21.2 0 41.3 8.4 56.5 23.5S416 154.8 416 176v160c0 21.2-8.4 41.3-23.5 56.5S357.2 416 336 416H176c-21.2 0-41.3-8.4-56.5-23.5S96 357.2 96 336V176c0-21.2 8.4-41.3 23.5-56.5S154.8 96 176 96h160m0-32H176c-61.6 0-112 50.4-112 112v160c0 61.6 50.4 112 112 112h160c61.6 0 112-50.4 112-112V176c0-61.6-50.4-112-112-112z"/><path d="M360 176c-13.3 0-24-10.7-24-24s10.7-24 24-24c13.2 0 24 10.7 24 24s-10.8 24-24 24zM256 192c35.3 0 64 28.7 64 64s-28.7 64-64 64-64-28.7-64-64 28.7-64 64-64m0-32c-53 0-96 43-96 96s43 96 96 96 96-43 96-96-43-96-96-96z"/>
        </svg>`,
    },
    {
      name: "LinkedIn",
      url: "https://www.linkedin.com/in/johnsutor3/",
      icon:
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
            <path d="M417.2 64H96.8C79.3 64 64 76.6 64 93.9V415c0 17.4 15.3 32.9 32.8 32.9h320.3c17.6 0 30.8-15.6 30.8-32.9V93.9C448 76.6 434.7 64 417.2 64zM183 384h-55V213h55v171zm-25.6-197h-.4c-17.6 0-29-13.1-29-29.5 0-16.7 11.7-29.5 29.7-29.5s29 12.7 29.4 29.5c0 16.4-11.4 29.5-29.7 29.5zM384 384h-55v-93.5c0-22.4-8-37.7-27.9-37.7-15.2 0-24.2 10.3-28.2 20.3-1.5 3.6-1.9 8.5-1.9 13.5V384h-55V213h55v23.8c8-11.4 20.5-27.8 49.6-27.8 36.1 0 63.4 23.8 63.4 75.1V384z"/>
        </svg>`
    },
    {
      name: "Resume",
      url: "https://johnsutor.com/resume_john_sutor.pdf",
      icon:
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
            <path d="M312 155h91c2.8 0 5-2.2 5-5 0-8.9-3.9-17.3-10.7-22.9L321 63.5c-5.8-4.8-13-7.4-20.6-7.4-4.1 0-7.4 3.3-7.4 7.4V136c0 10.5 8.5 19 19 19z"/><path d="M267 136V56H136c-17.6 0-32 14.4-32 32v336c0 17.6 14.4 32 32 32h240c17.6 0 32-14.4 32-32V181h-96c-24.8 0-45-20.2-45-45z"/>
        </svg>`
    },
    {
      name: "Email",
      url: "mailto:johnsutor3@gmail.com",
      icon:
        `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
            <path d="M460.6 147.3L353 256.9c-.8.8-.8 2 0 2.8l75.3 80.2c5.1 5.1 5.1 13.3 0 18.4-2.5 2.5-5.9 3.8-9.2 3.8s-6.7-1.3-9.2-3.8l-75-79.9c-.8-.8-2.1-.8-2.9 0L313.7 297c-15.3 15.5-35.6 24.1-57.4 24.2-22.1.1-43.1-9.2-58.6-24.9l-17.6-17.9c-.8-.8-2.1-.8-2.9 0l-75 79.9c-2.5 2.5-5.9 3.8-9.2 3.8s-6.7-1.3-9.2-3.8c-5.1-5.1-5.1-13.3 0-18.4l75.3-80.2c.7-.8.7-2 0-2.8L51.4 147.3c-1.3-1.3-3.4-.4-3.4 1.4V368c0 17.6 14.4 32 32 32h352c17.6 0 32-14.4 32-32V148.7c0-1.8-2.2-2.6-3.4-1.4z"/><path d="M256 295.1c14.8 0 28.7-5.8 39.1-16.4L452 119c-5.5-4.4-12.3-7-19.8-7H79.9c-7.5 0-14.4 2.6-19.8 7L217 278.7c10.3 10.5 24.2 16.4 39 16.4z"/>
        </svg>`
    }
  ])

  const [articles, setArticles] = useState([
    {
      title: '🌐 Dead Simple I18n Using Google Translate',
      description: `I18n (Internationalization) has never been simpler to implement using Google Sheets and Python.`,
      date: '2022-06-12T17:31:06+00:00',
      slug: 'dead-simple-i18n',
    },
    {
      title: '🦦 Do-it-yourself Jupyter Notebook Autograder with Otter Grader and Google Cloud',
      description: `With the expansion of SciTeens curriculum program, we decided to set out to create our own Jupyter Notebook autograder for our website.`,
      date: '2022-04-25T08:03:57+00:00',
      slug: 'diy-autograder-with-otter',
    },
    {
      title: '🔀 Replacing pre-trained Pytorch model layers with custom layers',
      description: `My contempt for Batch Normalization and love of Dropout layers led me to upgrade pre-trained convolutional neural networks in the simplest fashion possible.`,
      date: '2021-04-11T11:42:10+00:00',
      slug: 'replacing-pytorch-model-layers',
    }
  ])

  const [projects, setProjects] = useState([
    {
      title: "🧪 SciTeens",
      tags: ["SSR", "REST", "HTML", "CSS", "Firebase", "NextJS", "Google Cloud"],
      about: "The website for my nonprofit, SciTeens Inc. Built using NextJS with Server Side Rendering, with a Firebase backend, Google Kubernetes hosting, Google Functions, and OAuth 2.0for backend operations, and search powered by Algolia. Website is actively maintained.",
      url: "https://github.com/Sci-Teens/sciteens"
    },
    {
      title: "👨‍💻 Leetcode Study Tool",
      tags: ["Leetcode", "Python"],
      about: "This package provides a command-line tool for interracting with Leetcode to create flashcards for study, which can then be imported into Anki.",
      url: "https://github.com/johnsutor/leetcode-study-tool"
    },
    {
      title: "📷 Leopardi (Class-based Synthetic Data Generator)",
      tags: ["Blender", "Synthetic Data", "Rendering", "Python"],
      about: "A class-based library for generating 3D rendered synthetic data. This library allows for a myriad of control over the specifics of the rendering engine, including but not limited to lens field of view, rendering angle, and mode by which to load a background and/or 3D model to be rendered. This library is also fully extensible, and can be installed via Pip.",
      url: "https://github.com/johnsutor/leopardi",
    },
    {
      title: "💻 Synthblend (Command-line Synthetic Data Generator)",
      tags: ["Blender", "Synthetic Data", "Python", "CLI"],
      about: "A simple-to-use command-line tool for quickly working with synthetic data. With a directory of background images and another with object files and accompanying textures, you can quickly generate thousands of synthetic images to supplement any dataset. This library interfaces with the Blender bpy library.",
      url: "https://github.com/johnsutor/synthblend"
    }
  ])

  return (
    <>
      {/* <Head>
        <script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.4/dist/katex.min.js" integrity="sha384-PwRUT/YqbnEjkZO0zZxNqcxACrXe+j766U2amXcgMg5457rve2Y7I6ZJSm2A0mS4" crossorigin="anonymous"></script>
      </Head> */}
      <Layout>
        <div className="flex flex-col items-center justify-center py-2">
          <main className="h-full w-full">
            {/* Top of the page */}
            <div className="flex flex-col lg:flex-row w-full h-full">
              <div className="flex flex-col justify-center items-left -mt-12 lg:pt-0 lg:w-1/2 text-center lg:text-left h-screen ">
                <h2 className="text-6xl font-extrabold">
                  Hey, I'm John Sutor 👋
                </h2>
                <span className="flex w-full justify-center lg:justify-start items-center text-2xl font-bold mt-4 ">
                  {
                    profiles.map(profile => (
                      <a target="_blank" href={profile.url} className="mr-4">
                        <div className="h-10 w-10 fill-current text-gray-700 transition-transform hover:text-black hover:scale-110" dangerouslySetInnerHTML={{ __html: profile.icon }}></div>
                      </a>
                    ))
                  }
                </span>

              </div>
              <img className="lg:w-1/2 object-contain" src="john.jpg" alt="John Hiking" />
            </div>

            {/* About */}
            <div className="w-full text-xl mt-8 max-w-prose mx-auto">
              <h3 className="text-4xl font-bold">
                About Me
              </h3>
              <p>
                Howdy! I'm a current first-year masters student studying Data Science at New
                York University. Previously, I studied Applied Mathematics and Computational
                Science at Florida State University. There, I conducted research under
                Professor Jonathan Adams on the topic of Computer Vision and Synthetic
                Data within the lab that me and some other undergraduates started, the&nbsp;
                <a href="https://mllab.cci.fsu.edu/" target="_blank" className="font-bold">
                  Mlab
                </a>. I'm also one of the co-founders of&nbsp;
                <a href="https://sciteens.org" target="_blank" className="font-bold">
                  SciTeens
                </a>
                . I'm really into meal prepping, herping, camping/hiking, and vinyl
                collecting; feel free to reach out if you are too!
              </p>
            </div>
            {/* Articles */}
            <div className="w-full text-xl mt-8 max-w-prose mx-auto">
              <h3 className="text-4xl font-bold">
                Articles
              </h3>

              {articles.map((article, i) => (
                <Link href={"/articles/" + article.slug} key={i}>
                  <div className="py-4 cursor-pointer border-b-2 group transform duration-150 hover:py-6">
                    <h4 className="font-semibold text-3xl">{article.title}</h4>
                    <p className="italic mt-1">{(new Date(article.date)).toLocaleString('en')}</p>
                    <p className='text-gray-700'>{article.description}</p>
                  </div>
                </Link>
              ))}
            </div>

            {/* Projects} */}
            < div className="w-full text-xl mt-8 max-w-prose mx-auto" >
              <h3 className="text-4xl font-bold">
                What I've Created
              </h3>
              {projects.map((project, i) => (
                <a href={project.url} key={i}>
                  <div className="py-4 cursor-pointer border-b-2 group transform duration-150 hover:py-6">
                    <h4 className="font-semibold text-3xl">{project.title}</h4>
                    <p className="italic mt-1">
                      {project.tags.map(((tag, j) => (
                        <span className="inline-block bg-gray-200 rounded-full px-3 py-1 text-sm font-semibold text-gray-700 mr-2 group-hover:text-white group-hover:bg-black">{tag}</span>
                      )))}
                    </p>
                    <p className='text-gray-700'>{project.about}</p>
                  </div>
                </a>
              ))}
            </div>
          </main>
        </div>
      </Layout>
    </>
  )
}
