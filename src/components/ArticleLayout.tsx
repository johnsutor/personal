import { FunctionComponent, ReactNode } from "react";
import Layout from "./Layout";

type Props = {
    meta: JSON
    children: ReactNode;
};

const ArticleLayout: FunctionComponent<Props> = ({ meta, children }) => {
    return (
        <Layout>
            <article className="mx-auto prose prose-stone prose-md mb-10 prose-pre:bg-white prose-pre:p-0 prose-code:rounded-md prose-code:max-w-none prose-code:w-full">
                <h1 className="text-4xl font-bold">{meta.title}</h1>
                <p className="italic text-sm">{(new Date(meta.date)).toLocaleString('en')}</p>
                <p className="italic">{meta.description}</p>
                {children}
            </article>
        </Layout>
    );
};

export default ArticleLayout;