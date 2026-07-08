/**
 * This file plays the role of "the brand's own storefront code" -- product
 * catalog, cart state, checkout. It has nothing to do with CartIQ itself; it
 * just calls window.CartIQ.track(...) at the moments PRD Feature 1 says a
 * real integration would (add to cart, checkout steps, purchase), exactly
 * like a brand's dev team would wire up the snippet on their own site.
 */
const PRODUCTS = [
  { id: 1, name: "Wireless Headphones", category: "electronics", price: 349, emoji: "🎧" },
  { id: 2, name: "Denim Jacket", category: "apparel", price: 89, emoji: "🧥" },
  { id: 3, name: "Espresso Machine", category: "home & kitchen", price: 259, emoji: "☕" },
  { id: 4, name: "Skincare Gift Set", category: "beauty", price: 65, emoji: "🧴" },
  { id: 5, name: "Yoga Mat Bundle", category: "sports", price: 45, emoji: "🧘" },
  { id: 6, name: "Hardcover Novel Set", category: "books", price: 28, emoji: "📚" },
  { id: 7, name: "Gourmet Snack Box", category: "grocery", price: 19, emoji: "🍫" },
  { id: 8, name: "Building Blocks Set", category: "toys", price: 35, emoji: "🧸" },
];

const CART_KEY = "cartiq_demo_cart";

function readCart() {
  try {
    return JSON.parse(localStorage.getItem(CART_KEY)) || [];
  } catch {
    return [];
  }
}

function writeCart(cart) {
  localStorage.setItem(CART_KEY, JSON.stringify(cart));
  updateCartBadge();
}

function cartLines() {
  return readCart()
    .map((line) => ({ ...line, product: PRODUCTS.find((p) => p.id === line.id) }))
    .filter((l) => l.product);
}

function cartTotal() {
  return cartLines().reduce((sum, l) => sum + l.product.price * l.qty, 0);
}

function cartItemCount() {
  return cartLines().reduce((sum, l) => sum + l.qty, 0);
}

function addToCart(id, qty = 1) {
  const cart = readCart();
  const existing = cart.find((l) => l.id === id);
  if (existing) existing.qty += qty;
  else cart.push({ id, qty });
  writeCart(cart);

  const product = PRODUCTS.find((p) => p.id === id);
  window.CartIQ.track("add_to_cart", { product_category: product.category });
}

function removeFromCart(id) {
  writeCart(readCart().filter((l) => l.id !== id));
  window.CartIQ.track("remove_from_cart");
}

function updateCartBadge() {
  document.querySelectorAll(".cart-badge").forEach((el) => (el.textContent = cartItemCount()));
}

async function completePurchase() {
  const orderValue = cartTotal();
  writeCart([]);
  localStorage.setItem("cartiq_returning", "1");
  await window.CartIQ.track("purchase", { order_value: orderValue });
  return orderValue;
}

// Exposes just enough for cartiq.js to read live cart state off the host
// page, the same way a real snippet would read the brand's cart object.
window.CartIQStore = {
  getCart() {
    return { value: cartTotal(), items: cartItemCount() };
  },
  getCategory() {
    return window.CURRENT_PRODUCT_CATEGORY || null;
  },
};

document.addEventListener("DOMContentLoaded", updateCartBadge);
