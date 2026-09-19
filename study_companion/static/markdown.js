/* The syllabus's Markdown subset. Raw HTML is always rendered as text. */
export function escapeHTML(value) {
  return String(value).replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
}

function linkURL(raw) {
  if (/^https?:\/\//.test(raw)) return raw;
  if (/^(README|SYLLABUS)\.md(?:#.*)?$/.test(raw)) return `/source/${raw}`;
  return null;
}

export function inline(text) {
  const pattern = /`([^`]+)`|\[([^\]]+)\]\(([^\s)]+)\)|\*\*([^*]+)\*\*|\*([^*\n]+)\*/g;
  let result = "";
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    result += escapeHTML(text.slice(last, match.index));
    if (match[1] !== undefined) result += `<code>${escapeHTML(match[1])}</code>`;
    else if (match[2] !== undefined) {
      const url = linkURL(match[3]);
      result += url ? `<a href="${escapeHTML(url)}" target="_blank" rel="noopener noreferrer">${escapeHTML(match[2])} ↗</a>` : escapeHTML(match[2]);
    } else if (match[4] !== undefined) result += `<strong>${inline(match[4])}</strong>`;
    else result += `<em>${escapeHTML(match[5])}</em>`;
    last = match.index + match[0].length;
  }
  return result + escapeHTML(text.slice(last));
}

export function markdown(source) {
  const lines = source.trim().split("\n");
  let html = "";
  let i = 0;
  const isBlock = line => /^(#{1,6} |[-*] |\d+\. |```|\|)/.test(line);
  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) { i++; continue; }
    if (line.startsWith("```")) {
      const code = [];
      i++;
      while (i < lines.length && !lines[i].startsWith("```")) code.push(lines[i++]);
      i++;
      html += `<pre><code>${escapeHTML(code.join("\n"))}</code></pre>`;
      continue;
    }
    const heading = line.match(/^(#{1,6}) (.+)/);
    if (heading) {
      const level = Math.min(heading[1].length + 1, 6);
      html += `<h${level}>${inline(heading[2])}</h${level}>`;
      i++;
      continue;
    }
    if (/^\|/.test(line) && /^\|[\s:|-]+\|$/.test(lines[i + 1] || "")) {
      const cells = value => value.trim().replace(/^\||\|$/g, "").split("|");
      html += `<div class="table-scroll"><table><thead><tr>${cells(line).map(c => `<th>${inline(c.trim())}</th>`).join("")}</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && /^\|/.test(lines[i])) {
        html += `<tr>${cells(lines[i++]).map(c => `<td>${inline(c.trim())}</td>`).join("")}</tr>`;
      }
      html += "</tbody></table></div>";
      continue;
    }
    const list = line.match(/^([-*]|\d+\.) (.*)/);
    if (list) {
      const ordered = /^\d/.test(list[1]);
      const tag = ordered ? "ol" : "ul";
      const itemPattern = ordered ? /^\d+\. (.*)/ : /^[-*] (.*)/;
      html += `<${tag}>`;
      while (i < lines.length && itemPattern.test(lines[i])) {
        let item = lines[i++].replace(itemPattern, "$1");
        while (i < lines.length && lines[i].trim() && !isBlock(lines[i])) item += " " + lines[i++].trim();
        html += `<li>${inline(item)}</li>`;
      }
      html += `</${tag}>`;
      continue;
    }
    let paragraph = lines[i++];
    while (i < lines.length && lines[i].trim() && !isBlock(lines[i])) paragraph += " " + lines[i++].trim();
    html += `<p>${inline(paragraph)}</p>`;
  }
  return html;
}
