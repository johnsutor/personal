import Layout from "../../components/Layout";
import matter from "gray-matter";

export default function Papers({ papers }) {
  const loaded_papers = papers.map((paper) => matter(paper)).map((paper) => paper.data)

  return (
    <Layout>
      <div className="w-full text-xl mt-8">
            <h2 className="text-4xl font-bold">
              Papers
            </h2>
              {loaded_papers.map((paper, i) => (
                <a href={paper.url} key={i}>
                  <div className="py-2 cursor-pointer border-b-2">
                    <h2 className="font-semibold text-2xl">{paper.title}</h2>
                    <p className="italic text-gray-700">{paper.conference}</p>
                    <p>{paper.abstract}</p>
                  </div>
                </a>
            ))}
          </div>
    </Layout>
  )
}


export async function getStaticProps() {
  const fs = require("fs");

  const paper_md = fs.readdirSync(`${process.cwd()}/content/papers`, "utf-8").filter((fn) => fn.endsWith(".md"));

  const papers = paper_md.map((paper) => {
    const path = `${process.cwd()}/content/papers/${paper}`;
    const rawContent = fs.readFileSync(path, {
      encoding: "utf-8",
    });

    return rawContent;
  });

  return {
    props: {
      papers: papers,
    },
  };
}