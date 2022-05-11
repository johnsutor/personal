import Layout from '../components/Layout'
import Link from 'next/link'
import matter from "gray-matter";

export default function Home({ articles, projects }) {
  const loaded_articles = articles.map((article) => matter(article)).map((article) => article.data)
  const loaded_projects = projects.map((project) => matter(project)).map((project) => project.data)

  return (
    <Layout>
      <div className="flex flex-col items-center justify-center py-2">
        <main className="h-full w-full">
          {/* Top of the page */}
          <div className="flex flex-col lg:flex-row w-full h-full">
            <div className="flex flex-col justify-center items-center -mt-12 lg:pt-0 lg:w-1/2 text-center lg:text-left h-screen ">
              <h2 className="text-6xl font-extrabold">
                Hey, I'm John Sutor 👋
              </h2>
              <span className="flex w-full justify-center lg:justify-start items-center text-2xl font-bold mt-4">
                <Link href="/papers">
                  <a>
                  <span className="mr-2 bg-gradient-to-r from-blue-700 to-green-700 p-2 text-white rounded-md cursor-pointer">
                    Papers
                  </span>
                  </a>
                </Link>
                <a href="mailto:john@sciteens.org" className="mx-2 text-transparent bg-clip-text bg-gradient-to-r from-green-700 to-blue-700">
                  Contact
                </a>
              </span>
            </div>
            <img className="lg:w-1/2 object-contain" src="john.jpg" alt="John Hiking" />
          </div>
          
          {/* About */}
          <div className="w-full text-xl mt-8">
            <h3 className="text-2xl font-bold">
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
          <div className="w-full text-xl mt-8">
            <h3 className="text-2xl font-bold">
              Articles
            </h3>
              {loaded_articles.map((article, i) => (
                <Link href={`/articles/${article.slug}`} key={i}>
                  <a>
                  <div className="py-2 cursor-pointer border-b-2">
                    <a className="font-semibold">{article.title}</a>
                    <p>{article.description}</p>
                  </div>
                  </a>
                </Link>
            ))}
          </div>

          {/* Projects} */}
          <div className="w-full text-xl mt-8">
            <h3 className="text-2xl font-bold">
              What I've Created
            </h3>
              {loaded_projects.map((project, i) => (
                <a href={project.url} key={i}>
                  <div className="py-2 cursor-pointer border-b-2">
                    <p className="font-semibold">{project.title}</p>
                    <p className="italic text-gray-600">
                      {project.tags.map(((tag, j) => (
                        <span>{tag}, </span>
                      )))}
                      </p>
                    <p>{project.about}</p>
                  </div>
                </a>
            ))}
          </div>
        </main>
      </div>
    </Layout>

  )
}

export async function getStaticProps() {
  const fs = require("fs");

  const article_md = fs.readdirSync(`${process.cwd()}/content/articles`, "utf-8").filter((fn) => fn.endsWith(".md"));
  const project_md = fs.readdirSync(`${process.cwd()}/content/projects`, "utf-8").filter((fn) => fn.endsWith(".md"));

  const articles = article_md.map((article) => {
    const path = `${process.cwd()}/content/articles/${article}`;
    const rawContent = fs.readFileSync(path, {
      encoding: "utf-8",
    });

    return rawContent;
  });

  const projects = project_md.map((project) => {
    const path = `${process.cwd()}/content/projects/${project}`;
    const rawContent = fs.readFileSync(path, {
      encoding: "utf-8",
    });

    return rawContent;
  });

  return {
    props: {
      articles: articles,
      projects: projects,
    },
  };
}