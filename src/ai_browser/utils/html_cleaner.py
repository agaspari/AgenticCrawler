"""
HTML cleaning and preprocessing utilities.

Uses BeautifulSoup to strip noise from raw HTML before sending it to the LLM.
This significantly reduces token usage and improves extraction quality by
removing scripts, styles, navbars, footers, and other non-content elements.

Usage:
    cleaner = HTMLCleaner()
    text = cleaner.clean(raw_html)
    links = cleaner.extract_links(raw_html, base_url="https://example.com")
"""

from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag


# Elements that rarely contain useful content and inflate token counts
_NOISE_TAGS = [
    "script",
    "style",
    "noscript",
    "svg",
    "iframe",
    "object",
    "embed",
    "link",
    "meta",
    "head",
]

# Semantic sections that are typically navigation/boilerplate
_BOILERPLATE_TAGS = [
    "nav",
    "footer",
    "header",
]


class HTMLCleaner:
    """
    Cleans raw HTML to produce LLM-friendly text.

    Configurable noise removal with options to keep or remove
    boilerplate sections (headers, navs, footers) depending on the task.

    Attributes:
        remove_boilerplate: If True, also strips nav/header/footer elements.
        parser: BeautifulSoup parser to use (default: "html.parser").
    """

    def __init__(
        self,
        remove_boilerplate: bool = True,
        parser: str = "html.parser",
    ):
        self.remove_boilerplate = remove_boilerplate
        self.parser = parser

    def _make_soup(self, html: str) -> BeautifulSoup:
        """Parse HTML into a BeautifulSoup object."""
        return BeautifulSoup(html, self.parser)

    def _strip_noise(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Remove script, style, and other non-content elements."""
        tags_to_remove = list(_NOISE_TAGS)
        if self.remove_boilerplate:
            tags_to_remove.extend(_BOILERPLATE_TAGS)

        for tag_name in tags_to_remove:
            for element in soup.find_all(tag_name):
                element.decompose()

        # Also remove HTML comments
        from bs4 import Comment
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        return soup

    def clean(self, html: str, selector: str | None = None) -> str:
        """
        Clean HTML and return readable text.

        Args:
            html: Raw HTML string.
            selector: Optional CSS selector to narrow extraction to a
                      specific section (e.g., "main", "#content", ".job-list").

        Returns:
            Cleaned, readable text with normalized whitespace.
        """
        soup = self._make_soup(html)
        self._strip_noise(soup)

        # Optionally narrow to a specific section
        if selector:
            target = soup.select_one(selector)
            if target:
                soup = BeautifulSoup(str(target), self.parser)

        # Get text with newlines between block elements
        text = soup.get_text(separator="\n", strip=True)

        # Normalize whitespace: collapse multiple blank lines
        lines = [line.strip() for line in text.splitlines()]
        lines = [line for line in lines if line]  # Remove empty lines
        return "\n".join(lines)

    def extract_links(
        self,
        html: str,
        base_url: str | None = None,
        selector: str | None = None,
    ) -> list[dict[str, str]]:
        """
        Extract all hyperlinks from the HTML.

        Args:
            html: Raw HTML string.
            base_url: Base URL for resolving relative links.
            selector: Optional CSS selector to narrow the search scope.

        Returns:
            List of dicts with 'text' and 'href' keys.
        """
        soup = self._make_soup(html)
        self._strip_noise(soup)

        if selector:
            container = soup.select_one(selector)
            if container:
                anchors = container.find_all("a", href=True)
            else:
                anchors = []
        else:
            anchors = soup.find_all("a", href=True)

        links = []
        for a in anchors:
            href = a["href"]
            if base_url:
                href = urljoin(base_url, href)

            text = a.get_text(strip=True)
            if href and not href.startswith(("javascript:", "mailto:", "#")):
                links.append({"text": text, "href": href})

        return links

    def extract_elements(
        self,
        html: str,
        selector: str,
        attributes: list[str] | None = None,
    ) -> list[dict[str, str]]:
        """
        Extract elements matching a CSS selector.

        Args:
            html: Raw HTML string.
            selector: CSS selector to match elements.
            attributes: Optional list of attributes to extract from each element.
                        If None, extracts text content only.

        Returns:
            List of dicts with 'text' and any requested attribute keys.
        """
        soup = self._make_soup(html)
        elements = soup.select(selector)

        results = []
        for el in elements:
            item: dict[str, str] = {"text": el.get_text(strip=True)}
            if attributes:
                for attr in attributes:
                    item[attr] = el.get(attr, "")
            results.append(item)

        return results

    def get_page_title(self, html: str) -> str:
        """Extract the page title from HTML."""
        soup = self._make_soup(html)
        title = soup.find("title")
        return title.get_text(strip=True) if title else ""

    def get_meta_description(self, html: str) -> str:
        """Extract the meta description from HTML."""
        soup = self._make_soup(html)
        meta = soup.find("meta", attrs={"name": "description"})
        if meta and isinstance(meta, Tag):
            return meta.get("content", "")
        return ""
