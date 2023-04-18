import { FunctionComponent, ReactNode } from "react";
import Layout from "./Layout";

type Props = {
    children: ReactNode;
};

const ArticleLayout: FunctionComponent<Props> = ({ meta, children }) => {
    return (
        <Layout>
            <article className="mx-auto prose prose-stone prose-md">
                <h1 className="text-4xl font-bold">{meta.title}</h1>
                <p className="italic text-sm">{(new Date(meta.date)).toLocaleString('en')}</p>
                <p className="italic">{meta.description}</p>
                {children}
            </article>
        </Layout>
    );
};

export default ArticleLayout;