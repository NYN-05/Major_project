export default function Tip({ children, text, as = "span" }) {
  const Tag = as;
  return (
    <Tag className="tip" tabIndex={0} aria-label={text}>
      {children}
      <span className="tip-txt" role="tooltip">
        {text}
      </span>
    </Tag>
  );
}