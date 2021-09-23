import Layout from "../../components/Layout";
import matter from "gray-matter";

export default function Press({ press }) {
  const loaded_articles = press.map((article) => matter(article)).map((article) => article.data)

  return (
    
    <Layout>
      <div className="w-full text-xl mt-8">
            <h2 className="text-4xl font-bold">
              Press
            </h2>
              {loaded_articles.map((article, i) => (
                <a href={article.url} key={i}>
                  <div className="py-2 flex flex-col lg:flex-row items-center my-4 border-b-2">
                    <div className="lg:w-1/4 object-contain">
                        <img src={article.image} className="w-full h-full object-contain lg:rounded-md"/>
                    </div>
                    <div className="mt-4 lg:ml-4 lg:w-3/4 px-2">
                        <h2 className="font-semibold">{article.title}</h2>
                        <p className="italic">{article.organization}</p>
                        <p>{article.snippet}</p>
                    </div>
                  </div>
                </a>
            ))}
          </div>

          
    </Layout>
  )
}


export async function getStaticProps() {
  const fs = require("fs");

  const press_md = fs.readdirSync(`${process.cwd()}/content/press`, "utf-8").filter((fn) => fn.endsWith(".md"));

  const press = press_md.map((article) => {
    const path = `${process.cwd()}/content/press/${article}`;
    const rawContent = fs.readFileSync(path, {
      encoding: "utf-8",
    });

    return rawContent;
  });

  return {
    props: {
      press: press,
    },
  };
}