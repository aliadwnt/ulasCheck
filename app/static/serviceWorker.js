self.addEventListener('install', event => {
  console.log('Service Worker installed');
});

self.addEventListener('fetch', event => {
  event.respondWith(
    fetch(event.request).catch(() => new Response("Offline mode aktif"))
  );
});
